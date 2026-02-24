from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

RULE_PATH = Path(__file__).resolve().parent.parent / "SupplierScoringRule.json"

DEFAULT_LINE_FIT_CONFIG: Dict[str, Any] = {
    "weights": {
        "price": 0.3,
        "lead_time": 0.2,
        "capacity": 0.15,
        "site": 0.1,
        "commercial": 0.1,
        "compliance": 0.1,
        "responsiveness": 0.05,
    },
    "constants": {
        "cap_gate_ratio": 0.8,
        "notice_ref_days": 30,
        "resp_ref_hrs": 48,
        "exception_ref": 10,
        "exception_weight": 0.1,
    },
    "exception_weights": {
        "penalty": 3,
        "liability": 5,
        "price_adjustment": 4,
        "price_reopener": 4,
        "unlimited_liability": 6,
    },
    "region_map": {
        "asia": ["China", "Vietnam", "South Korea", "Japan", "India"],
        "north_america": ["United States", "USA", "Canada", "Mexico"],
        "europe": ["Germany", "France", "Italy", "Netherlands", "UK", "United Kingdom"],
    },
}

DEFAULT_SUPPLIER_FIT_CONFIG: Dict[str, Any] = {
    "metric_weights": {
        "supplierTier": 15.0,
        "relationshipTenureMonths": 10.0,
        "otifPct12m": 20.0,
        "leadTimeMeanDays12m": 15.0,
        "defectPpm12m": 15.0,
        "majorIncidentCount12m": 10.0,
        "priceVariancePct12m": 10.0,
        "complianceCertLevel": 5.0,
    },
    "tier_scores": {
        "strategic": 100.0,
        "preferred": 85.0,
        "approved": 70.0,
        "new": 55.0,
        "default": 55.0,
    },
    "relationship_tenure_bands": [
        {"min": 36, "score": 100.0},
        {"min": 12, "score": 80.0},
        {"min": 3, "score": 60.0},
        {"score": 50.0},
    ],
    "otif_bands": [
        {"min": 98, "score": 100.0},
        {"min": 95, "score": 90.0},
        {"min": 90, "score": 75.0},
        {"min": 80, "score": 55.0},
        {"score": 30.0},
    ],
    "lead_time_mean_bands": [
        {"max": 14, "score": 100.0},
        {"max": 30, "score": 80.0},
        {"max": 45, "score": 60.0},
        {"max": 60, "score": 40.0},
        {"score": 20.0},
    ],
    "defect_ppm_bands": [
        {"max": 50, "score": 100.0},
        {"max": 200, "score": 85.0},
        {"max": 500, "score": 70.0},
        {"max": 1000, "score": 50.0},
        {"score": 25.0},
    ],
    "major_incident_bands": [
        {"max": 0, "score": 100.0},
        {"max": 1, "score": 70.0},
        {"max": 2, "score": 40.0},
        {"score": 0.0},
    ],
    "price_variance_bands": [
        {"max": 2, "score": 100.0},
        {"max": 5, "score": 85.0},
        {"max": 10, "score": 70.0},
        {"max": 15, "score": 50.0},
        {"score": 30.0},
    ],
    "compliance_scores": {
        "none": 40.0,
        "unknown": 40.0,
        "iso9001": 70.0,
        "iso14001": 80.0,
        "esg": 80.0,
        "iatf16949": 90.0,
        "multi_core": 100.0,
        "default": 40.0,
    },
}

DEFAULT_FINAL_LINE_CONFIG: Dict[str, Any] = {
    "line_fit_weight": 1.0,
    "supplier_fit_weight": 1.0,
    "normalization": {
        "method": "divide_by_max_raw",
        "max_raw": 200.0,
        "output_scale": 100.0,
    },
}


def _load_rule() -> Dict[str, Any]:
    return json.loads(RULE_PATH.read_text(encoding="utf-8"))


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_number_with_error(
    value: Any,
    label: str,
    errors: Optional[List[str]] = None,
) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if errors is not None:
            errors.append(f"Invalid {label} value: {value}")
        return None


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _parse_iso_datetime(value: Optional[str], errors: Optional[List[str]] = None, label: str = "datetime") -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        if errors is not None:
            errors.append(f"Invalid {label} value: {value}")
        return None


def _select_base_tier(volume_tiers: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
    if not volume_tiers:
        return None, None
    sorted_tiers = sorted(
        volume_tiers,
        key=lambda tier: _safe_number(tier.get("thresholdQty")) or float("inf"),
    )
    base_tier = sorted_tiers[0]
    return base_tier.get("tierName"), _safe_number(base_tier.get("thresholdQty"))


def _parse_net_days(
    payment_terms: Optional[str],
    errors: Optional[List[str]] = None,
    label: str = "paymentTerms",
) -> Optional[float]:
    if not payment_terms:
        return None
    match = re.search(r"(\d+)", payment_terms)
    if not match:
        if errors is not None:
            errors.append(f"Invalid {label} value: {payment_terms}")
        return None
    return float(match.group(1))


def _weighted_sum(values: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(values.get(key, 0.0) * weight for key, weight in weights.items())


def _weighted_average(values: Dict[str, Optional[float]], weights: Dict[str, float]) -> float:
    total = 0.0
    weight_total = 0.0
    for key, value in values.items():
        if value is None:
            continue
        weight = weights.get(key, 0.0)
        if weight <= 0:
            continue
        total += value * weight
        weight_total += weight
    if weight_total == 0:
        return 0.0
    return total / weight_total


def _normalize_supplier_ids(supplier_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if not supplier_ids:
        return None
    return {str(supplier_id) for supplier_id in supplier_ids if supplier_id}


def _region_for_country(country: str, region_map: Dict[str, str]) -> Optional[str]:
    normalized = country.strip().lower()
    for region, countries in region_map.items():
        if normalized in {item.lower() for item in countries}:
            return region
    return None


def _to_percentage(value: Any) -> Optional[float]:
    number = _safe_number(value)
    if number is None:
        return None
    return number * 100 if -1 <= number <= 1 else number


def _scenario_multiplier_for_line(
    scenario_context: Dict[str, Any],
    line_index: int,
) -> float:
    multipliers = scenario_context.get("demand_multipliers")
    if not isinstance(multipliers, list) or not multipliers:
        return 1.0
    raw = _safe_number(multipliers[line_index % len(multipliers)])
    if raw is None or raw <= 0:
        return 1.0
    return raw


def _score_from_bands(
    value: Optional[float],
    bands: List[Dict[str, Any]],
    default_score: float = 0.0,
) -> float:
    if value is None:
        return default_score
    for band in bands:
        min_value = _safe_number(band.get("min"))
        max_value = _safe_number(band.get("max"))
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        score = _safe_number(band.get("score"))
        if score is not None:
            return _clamp(score, 0.0, 100.0)
    return _clamp(default_score, 0.0, 100.0)


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _build_final_line_score(
    line_fit_score: float,
    supplier_fit_score: float,
    policy: Dict[str, Any],
) -> Tuple[float, float]:
    line_weight = _safe_number(policy.get("line_fit_weight"))
    supplier_weight = _safe_number(policy.get("supplier_fit_weight"))
    if line_weight is None:
        line_weight = float(DEFAULT_FINAL_LINE_CONFIG["line_fit_weight"])
    if supplier_weight is None:
        supplier_weight = float(DEFAULT_FINAL_LINE_CONFIG["supplier_fit_weight"])
    line_weight = max(0.0, line_weight)
    supplier_weight = max(0.0, supplier_weight)
    if line_weight == 0 and supplier_weight == 0:
        line_weight = float(DEFAULT_FINAL_LINE_CONFIG["line_fit_weight"])
        supplier_weight = float(DEFAULT_FINAL_LINE_CONFIG["supplier_fit_weight"])

    raw_score = (line_fit_score * line_weight) + (supplier_fit_score * supplier_weight)

    normalization = policy.get("normalization") or DEFAULT_FINAL_LINE_CONFIG["normalization"]
    method = str(normalization.get("method") or "divide_by_max_raw").strip().lower()
    output_scale = _safe_number(normalization.get("output_scale"))
    if output_scale is None or output_scale <= 0:
        output_scale = float(DEFAULT_FINAL_LINE_CONFIG["normalization"]["output_scale"])

    if method == "clamp_from_100":
        final_score = _clamp(raw_score / 100.0, 0.0, 1.0) * output_scale
    elif method == "none":
        final_score = _clamp(raw_score, 0.0, output_scale)
    else:
        max_raw = _safe_number(normalization.get("max_raw"))
        if max_raw is None or max_raw <= 0:
            max_raw = max((100 * line_weight) + (100 * supplier_weight), 1.0)
        final_score = _clamp(raw_score / max_raw, 0.0, 1.0) * output_scale

    return raw_score, final_score


def _compliance_level_score(level: Any, compliance_scores: Dict[str, float]) -> float:
    normalized = _normalize_key(level)
    if not normalized:
        return float(compliance_scores.get("unknown", compliance_scores.get("default", 40.0)))
    if normalized in {"none", "unknown", "na", "n/a"}:
        return float(compliance_scores.get("none", compliance_scores.get("default", 40.0)))

    found_levels: set[str] = set()
    if "iso9001" in normalized:
        found_levels.add("iso9001")
    if "iso14001" in normalized:
        found_levels.add("iso14001")
    if "iatf16949" in normalized:
        found_levels.add("iatf16949")
    if "esg" in normalized:
        found_levels.add("esg")

    if len(found_levels) >= 2 and "multi_core" in compliance_scores:
        return float(compliance_scores["multi_core"])
    for key in ("iatf16949", "iso14001", "esg", "iso9001"):
        if key in found_levels and key in compliance_scores:
            return float(compliance_scores[key])
    return float(compliance_scores.get("default", 40.0))


def _supplier_fit_score(
    supplier: Optional[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    metric_weights = policy.get("metric_weights") or DEFAULT_SUPPLIER_FIT_CONFIG["metric_weights"]
    tier_scores = policy.get("tier_scores") or DEFAULT_SUPPLIER_FIT_CONFIG["tier_scores"]
    compliance_scores = policy.get("compliance_scores") or DEFAULT_SUPPLIER_FIT_CONFIG["compliance_scores"]

    if not supplier:
        return 0.0, {
            "inputs": {},
            "components": {},
            "weights": metric_weights,
            "penalties": [],
            "weightedTotal": 0.0,
        }

    supplier_tier_key = _normalize_key(supplier.get("supplierTier"))
    supplier_tier_score = float(tier_scores.get(supplier_tier_key, tier_scores.get("default", 55.0)))
    relationship_score = _score_from_bands(
        _safe_number(supplier.get("relationshipTenureMonths")),
        policy.get("relationship_tenure_bands") or DEFAULT_SUPPLIER_FIT_CONFIG["relationship_tenure_bands"],
        50.0,
    )
    otif_score = _score_from_bands(
        _to_percentage(supplier.get("otifPct12m")),
        policy.get("otif_bands") or DEFAULT_SUPPLIER_FIT_CONFIG["otif_bands"],
        30.0,
    )
    lead_time_mean_score = _score_from_bands(
        _safe_number(supplier.get("leadTimeMeanDays12m")),
        policy.get("lead_time_mean_bands") or DEFAULT_SUPPLIER_FIT_CONFIG["lead_time_mean_bands"],
        20.0,
    )
    defect_score = _score_from_bands(
        _safe_number(supplier.get("defectPpm12m")),
        policy.get("defect_ppm_bands") or DEFAULT_SUPPLIER_FIT_CONFIG["defect_ppm_bands"],
        25.0,
    )
    major_incident_score = _score_from_bands(
        _safe_number(supplier.get("majorIncidentCount12m")),
        policy.get("major_incident_bands") or DEFAULT_SUPPLIER_FIT_CONFIG["major_incident_bands"],
        0.0,
    )
    price_variance_score = _score_from_bands(
        _to_percentage(supplier.get("priceVariancePct12m")),
        policy.get("price_variance_bands") or DEFAULT_SUPPLIER_FIT_CONFIG["price_variance_bands"],
        30.0,
    )
    compliance_score = _clamp(_compliance_level_score(supplier.get("complianceCertLevel"), compliance_scores), 0.0, 100.0)

    components = {
        "supplierTier": supplier_tier_score,
        "relationshipTenureMonths": relationship_score,
        "otifPct12m": otif_score,
        "leadTimeMeanDays12m": lead_time_mean_score,
        "defectPpm12m": defect_score,
        "majorIncidentCount12m": major_incident_score,
        "priceVariancePct12m": price_variance_score,
        "complianceCertLevel": compliance_score,
    }

    active_weights = {
        metric_key: float(weight)
        for metric_key, weight in metric_weights.items()
        if metric_key in components and float(weight) > 0
    }
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        weighted_total = 0.0
    else:
        weighted_total = sum(
            float(components.get(metric_key, 0.0)) * weight
            for metric_key, weight in active_weights.items()
        ) / total_weight

    penalties: List[Dict[str, Any]] = []
    supplier_status_key = _normalize_key(supplier.get("supplierStatus"))
    if supplier_status_key in {"blocked", "inactive"}:
        penalties.append({"id": "supplier_status_gate", "delta": -100.0})
        weighted_total = 0.0
    if supplier.get("msaActiveFlag") is False:
        penalties.append({"id": "msa_active_gate", "delta": -100.0})
        weighted_total = 0.0

    final_score = _clamp(weighted_total, 0.0, 100.0)
    breakdown = {
        "inputs": {
            "supplierStatus": supplier.get("supplierStatus"),
            "supplierTier": supplier.get("supplierTier"),
            "relationshipTenureMonths": supplier.get("relationshipTenureMonths"),
            "otifPct12m": supplier.get("otifPct12m"),
            "leadTimeMeanDays12m": supplier.get("leadTimeMeanDays12m"),
            "defectPpm12m": supplier.get("defectPpm12m"),
            "majorIncidentCount12m": supplier.get("majorIncidentCount12m"),
            "priceVariancePct12m": supplier.get("priceVariancePct12m"),
            "complianceCertLevel": supplier.get("complianceCertLevel"),
            "msaActiveFlag": supplier.get("msaActiveFlag"),
        },
        "components": components,
        "weights": active_weights,
        "penalties": penalties,
        "weightedTotal": weighted_total,
    }
    return final_score, breakdown


def _pick_line_price(
    tier_prices: List[Dict[str, Any]],
    base_tier_name: Optional[str],
    errors: Optional[List[str]] = None,
    line_id: Optional[str] = None,
) -> Optional[float]:
    if not tier_prices:
        return None
    matched = next((entry for entry in tier_prices if entry.get("tierName") == base_tier_name), None)
    if matched and matched.get("unitPrice") is not None:
        label = f"unitPrice ({line_id})" if line_id else "unitPrice"
        return _safe_number_with_error(matched.get("unitPrice"), label, errors)
    for entry in tier_prices:
        if entry.get("unitPrice") is not None:
            label = f"unitPrice ({line_id})" if line_id else "unitPrice"
            return _safe_number_with_error(entry.get("unitPrice"), label, errors)
    return None


def score_suppliers(
    rfq: Dict[str, Any],
    response_details: List[Dict[str, Any]],
    supplier_master: Optional[List[Dict[str, Any]]] = None,
    supplier_ids: Optional[Iterable[str]] = None,
    include_details: bool = False,
    scenario_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rule = _load_rule()
    score_decimals = rule.get("engine", {}).get("rounding", {}).get("score_decimals", 1)
    policy_config = rule.get("policy", {})
    line_config = policy_config.get("lineFit", {})
    supplier_config = policy_config.get("supplierFit", {})
    final_line_config = {
        **DEFAULT_FINAL_LINE_CONFIG,
        **(policy_config.get("finalLine") or {}),
    }
    final_line_config["normalization"] = {
        **DEFAULT_FINAL_LINE_CONFIG["normalization"],
        **((policy_config.get("finalLine") or {}).get("normalization") or {}),
    }

    line_weights = line_config.get(
        "weights",
        DEFAULT_LINE_FIT_CONFIG["weights"],
    )
    line_constants = {
        **DEFAULT_LINE_FIT_CONFIG["constants"],
        **(line_config.get("constants") or {}),
    }
    exception_weights = {
        **DEFAULT_LINE_FIT_CONFIG["exception_weights"],
        **(line_config.get("exception_weights") or {}),
    }
    region_map = line_config.get(
        "region_map",
        DEFAULT_LINE_FIT_CONFIG["region_map"],
    )

    supplier_fit_policy = {
        **DEFAULT_SUPPLIER_FIT_CONFIG,
        **supplier_config,
    }
    supplier_fit_policy["metric_weights"] = supplier_config.get(
        "metric_weights",
        DEFAULT_SUPPLIER_FIT_CONFIG["metric_weights"],
    )
    supplier_fit_policy["tier_scores"] = {
        **DEFAULT_SUPPLIER_FIT_CONFIG["tier_scores"],
        **(supplier_config.get("tier_scores") or {}),
    }
    supplier_fit_policy["compliance_scores"] = {
        **DEFAULT_SUPPLIER_FIT_CONFIG["compliance_scores"],
        **(supplier_config.get("compliance_scores") or {}),
    }

    rfq_detail = rfq.get("detail") or {}
    rfq_cover = rfq_detail.get("cover") or {}
    rfq_lines = rfq_detail.get("rfqLines") or []
    rfq_plants = rfq_detail.get("rfqPlants") or []
    base_tier_name, base_tier_qty = _select_base_tier(rfq_detail.get("rfqVolumeTiers") or [])
    required_monthly_qty = base_tier_qty or 1.0
    deadline_value = rfq_cover.get("quoteDeadlineUtc")

    response_map = {entry["supplierId"]: entry for entry in response_details}
    supplier_master_map = {entry["id"]: entry for entry in supplier_master or []}

    all_suppliers = rfq.get("suppliers") or []
    supplier_filter = _normalize_supplier_ids(supplier_ids)
    suppliers = [supplier for supplier in all_suppliers if not supplier_filter or supplier["id"] in supplier_filter]

    context = scenario_context or {}
    scenario_cost_multiplier = _safe_number(context.get("cost_multiplier"))
    if scenario_cost_multiplier is None or scenario_cost_multiplier <= 0:
        scenario_cost_multiplier = 1.0
    scenario_score_multiplier = _safe_number(context.get("score_multiplier"))
    if scenario_score_multiplier is None or scenario_score_multiplier <= 0:
        scenario_score_multiplier = 1.0
    scenario_exception_cost_multiplier = _safe_number(context.get("exception_cost_multiplier"))
    if scenario_exception_cost_multiplier is None or scenario_exception_cost_multiplier <= 0:
        scenario_exception_cost_multiplier = 1.0
    scenario_exception_score_multiplier = _safe_number(context.get("exception_score_multiplier"))
    if scenario_exception_score_multiplier is None or scenario_exception_score_multiplier <= 0:
        scenario_exception_score_multiplier = 1.0
    scenario_compliance_score_multiplier = _safe_number(context.get("compliance_score_multiplier"))
    if scenario_compliance_score_multiplier is None or scenario_compliance_score_multiplier <= 0:
        scenario_compliance_score_multiplier = 1.0
    capacity_shock_pct = _safe_number(context.get("capacity_shock_pct"))
    if capacity_shock_pct is None or capacity_shock_pct <= 0:
        capacity_shock_pct = 1.0
    capacity_shock_supplier_id = str(context.get("capacity_shock_supplier_id") or "")

    line_price_matrix: Dict[str, Dict[str, Optional[float]]] = {}
    line_input_errors: Dict[str, Dict[str, List[str]]] = {}
    line_lt_values: Dict[str, List[float]] = {line["rfqLineId"]: [] for line in rfq_lines}
    line_net_values: Dict[str, List[float]] = {line["rfqLineId"]: [] for line in rfq_lines}
    line_price_values: Dict[str, List[float]] = {line["rfqLineId"]: [] for line in rfq_lines}

    for supplier in all_suppliers:
        supplier_id = supplier["id"]
        response = response_map.get(supplier_id, {})
        quote_header = response.get("quoteHeader") or {}
        quote_lines = response.get("quoteLines") or []
        tier_prices = response.get("quoteLineTierPrices") or []
        exception_tags = {tag.get("tag") for tag in (response.get("quoteExceptionTags") or []) if tag.get("tag")}

        line_id_map = {line.get("rfqLineId"): line.get("quoteLineId") for line in quote_lines}
        line_price_matrix[supplier_id] = {}
        line_input_errors[supplier_id] = {}

        for line in rfq_lines:
            rfq_line_id = line.get("rfqLineId")
            quote_line_id = line_id_map.get(rfq_line_id)
            if not quote_line_id:
                line_price_matrix[supplier_id][rfq_line_id] = None
                line_input_errors[supplier_id][rfq_line_id] = []
                continue
            prices = [entry for entry in tier_prices if entry.get("quoteLineId") == quote_line_id]
            line_input_errors[supplier_id][rfq_line_id] = []
            unit_price = _pick_line_price(
                prices,
                base_tier_name,
                line_input_errors[supplier_id][rfq_line_id],
                rfq_line_id,
            )
            if unit_price is not None:
                unit_price *= scenario_cost_multiplier
                if exception_tags.intersection({"price_adjustment", "price_reopener"}):
                    unit_price *= scenario_exception_cost_multiplier
            line_price_matrix[supplier_id][rfq_line_id] = unit_price
            if unit_price is not None:
                line_price_values[rfq_line_id].append(unit_price)
                lt_value = _safe_number(quote_header.get("e2eLeadTimeDays") or quote_header.get("stdProdLeadTimeDays"))
                if lt_value is not None:
                    line_lt_values[rfq_line_id].append(lt_value)
                net_days = _parse_net_days(quote_header.get("paymentTerms"))
                if net_days is not None:
                    line_net_values[rfq_line_id].append(net_days)

    line_stats = {}
    for line in rfq_lines:
        rfq_line_id = line.get("rfqLineId")
        prices = line_price_values.get(rfq_line_id, [])
        lts = line_lt_values.get(rfq_line_id, [])
        nets = line_net_values.get(rfq_line_id, [])
        line_stats[rfq_line_id] = {
            "price_min": min(prices) if prices else None,
            "price_max": max(prices) if prices else None,
            "lt_min": min(lts) if lts else None,
            "lt_max": max(lts) if lts else None,
            "net_min": min(nets) if nets else None,
            "net_max": max(nets) if nets else None,
        }

    line_weight_value = 1 / len(rfq_lines) if rfq_lines else 1
    line_weights_map = {line.get("rfqLineId"): line_weight_value for line in rfq_lines}

    scores: List[Dict[str, Any]] = []
    for supplier in suppliers:
        supplier_id = supplier["id"]
        response = response_map.get(supplier_id, {})
        quote_header = response.get("quoteHeader") or {}
        exceptions = response.get("quoteExceptionTags") or []
        exception_tags = [tag.get("tag") for tag in exceptions if tag.get("tag")]
        exception_tag_set = set(exception_tags)
        manufacturing_sites = response.get("quoteManufacturingSites") or []
        supplier_master_entry = supplier_master_map.get(supplier_id)

        supplier_fit_score, supplier_fit_breakdown = _supplier_fit_score(
            supplier_master_entry,
            supplier_fit_policy,
        )

        line_scores: List[Dict[str, Any]] = []
        event_weighted_total = 0.0
        event_weight_total = 0.0
        total_cost = 0.0
        event_errors: List[str] = []
        for line_index, line in enumerate(rfq_lines):
            rfq_line_id = line.get("rfqLineId")
            line_errors = list(line_input_errors.get(supplier_id, {}).get(rfq_line_id, []))
            line_stats_entry = line_stats.get(rfq_line_id, {})
            unit_price = line_price_matrix.get(supplier_id, {}).get(rfq_line_id)
            price_min = line_stats_entry.get("price_min")
            price_max = line_stats_entry.get("price_max")
            demand_multiplier = _scenario_multiplier_for_line(context, line_index)
            line_required_monthly_qty = required_monthly_qty * demand_multiplier
            if unit_price is not None:
                total_cost += unit_price * line_required_monthly_qty

            lt_raw = quote_header.get("e2eLeadTimeDays")
            if lt_raw is None:
                lt_raw = quote_header.get("stdProdLeadTimeDays")
            lt_value = _safe_number_with_error(lt_raw, "leadTimeDays", line_errors)
            lt_min = line_stats_entry.get("lt_min")
            lt_max = line_stats_entry.get("lt_max")

            net_days = _parse_net_days(quote_header.get("paymentTerms"), line_errors)
            net_min = line_stats_entry.get("net_min")
            net_max = line_stats_entry.get("net_max")

            gates = {
                "validity": None,
                "dg_compliance": None,
                "capacity": None,
            }
            min_validity = _safe_number_with_error(rfq_cover.get("minQuoteValidityDays"), "minQuoteValidityDays", line_errors)
            quote_validity = _safe_number_with_error(quote_header.get("quoteValidityDays"), "quoteValidityDays", line_errors)
            if min_validity is None or quote_validity is None:
                gates["validity"] = None
            else:
                gates["validity"] = quote_validity >= min_validity

            is_dg = quote_header.get("isDG") is True
            dg_ok = True
            if is_dg:
                dg_ok = bool(quote_header.get("un38_3Provided")) and bool(quote_header.get("dgPackingSupported"))
            gates["dg_compliance"] = dg_ok

            capacity_qty = _safe_number_with_error(quote_header.get("capacityMonthlyQty"), "capacityMonthlyQty", line_errors)
            if capacity_qty is not None and supplier_id == capacity_shock_supplier_id:
                capacity_qty *= capacity_shock_pct
            if capacity_qty is None:
                gates["capacity"] = None
            else:
                gates["capacity"] = capacity_qty >= line_required_monthly_qty * line_constants["cap_gate_ratio"]

            # DG compliance is treated as a soft scoring factor (compliance sub-score),
            # while validity/capacity remain hard gates that can force zero line score.
            hard_gate_keys = ("validity", "capacity")
            gate_failed = any(gates.get(key) is False for key in hard_gate_keys)
            if gate_failed:
                line_fit_score = 0.0
                base_line_score, _ = _build_final_line_score(
                    line_fit_score,
                    supplier_fit_score,
                    final_line_config,
                )
                final_line_score = 0.0
                line_scores.append(
                    {
                        "rfqLineId": rfq_line_id,
                        "lineNo": line.get("lineNo"),
                        "lineFitScore": line_fit_score,
                        "supplierFitScore": supplier_fit_score,
                        "baseLineScore": base_line_score,
                        "finalLineScore": final_line_score,
                        "subScores": {},
                        "penalties": {},
                        "gates": gates,
                        "inputs": {
                            "unitPrice": unit_price,
                            "leadTime": lt_value,
                            "netDays": net_days,
                            "requiredMonthlyQty": line_required_monthly_qty,
                            "demandMultiplier": demand_multiplier,
                        },
                        "errors": line_errors,
                    }
                )
                if line_errors:
                    event_errors.extend([f"{rfq_line_id}: {error}" for error in line_errors])
                event_weighted_total += final_line_score * line_weights_map.get(rfq_line_id, 0)
                event_weight_total += line_weights_map.get(rfq_line_id, 0)
                continue

            if unit_price is None or price_min is None or price_max is None:
                s_price = None
            elif price_max == price_min:
                s_price = 1.0
            else:
                s_price = _clamp((price_max - unit_price) / (price_max - price_min))

            if lt_value is None or lt_min is None or lt_max is None:
                s_lt_base = None
            elif lt_max == lt_min:
                s_lt_base = 1.0
            else:
                s_lt_base = _clamp((lt_max - lt_value) / (lt_max - lt_min))

            expedite_available = quote_header.get("expediteAvailable") is True
            expedite_lt = _safe_number_with_error(
                quote_header.get("expediteLeadTimeDays"), "expediteLeadTimeDays", line_errors
            )
            expedite_premium_raw = quote_header.get("expeditePremiumPct")
            expedite_premium = _safe_number_with_error(expedite_premium_raw, "expeditePremiumPct", line_errors)
            if expedite_premium is None and expedite_premium_raw is None:
                expedite_premium = 0.0
            if expedite_available and expedite_lt and lt_value and expedite_premium is not None:
                bonus_raw = _clamp((lt_value - expedite_lt) / lt_value)
                bonus_exp = bonus_raw * (1 - expedite_premium / 100)
            else:
                bonus_exp = 0.0
            if s_lt_base is None:
                s_lt = None
            else:
                s_lt = 0.8 * s_lt_base + 0.2 * bonus_exp

            flex_pct_raw = quote_header.get("capacityFlexPct")
            flex_pct = _safe_number_with_error(flex_pct_raw, "capacityFlexPct", line_errors)
            if flex_pct is None and flex_pct_raw is None:
                flex_pct = 0.0
            flex_notice_raw = quote_header.get("capacityFlexNoticeDays")
            flex_notice = _safe_number_with_error(flex_notice_raw, "capacityFlexNoticeDays", line_errors)
            if flex_notice is None and flex_notice_raw is None:
                flex_notice = line_constants["notice_ref_days"]
            if capacity_qty is None or flex_pct is None or flex_notice is None or not line_required_monthly_qty:
                s_cap = None
            else:
                coverage = (capacity_qty * (1 + flex_pct / 100)) / line_required_monthly_qty
                s_cap_qty = _clamp(coverage)
                s_cap_notice = _clamp(1 - (flex_notice / line_constants["notice_ref_days"]))
                s_cap = s_cap_qty * s_cap_notice

            site_countries = {site.get("country") for site in manufacturing_sites if site.get("country")}
            plant_countries = {plant.get("country") for plant in rfq_plants if plant.get("country")}
            match_country = bool(site_countries.intersection(plant_countries))
            match_region = False
            if not match_country:
                for site_country in site_countries:
                    site_region = _region_for_country(site_country, region_map)
                    if not site_region:
                        continue
                    for plant_country in plant_countries:
                        if _region_for_country(plant_country, region_map) == site_region:
                            match_region = True
                            break
                    if match_region:
                        break
            if match_country:
                s_site = 1.0
            elif match_region:
                s_site = 0.5
            else:
                s_site = 0.0

            if min_validity is None or quote_validity is None:
                s_valid = None
            else:
                s_valid = _clamp(quote_validity / min_validity) if min_validity else 0.0
            moq_raw = quote_header.get("moqQty")
            moq_qty = _safe_number_with_error(moq_raw, "moqQty", line_errors)
            if moq_qty is None and moq_raw is None:
                s_moq = 1.0
            elif moq_qty is None:
                s_moq = None
            else:
                s_moq = _clamp(line_required_monthly_qty / moq_qty) if moq_qty else 0.0
            mpq_raw = quote_header.get("mpqQty")
            mpq_qty = _safe_number_with_error(mpq_raw, "mpqQty", line_errors)
            if mpq_qty is None and mpq_raw is None:
                s_mpq = 1.0
            elif mpq_qty is None:
                s_mpq = None
            else:
                s_mpq = _clamp(line_required_monthly_qty / mpq_qty) if mpq_qty else 0.0
            if net_days is None or net_min is None or net_max is None:
                s_pay = None
            elif net_max == net_min:
                s_pay = 1.0
            else:
                s_pay = _clamp((net_days - net_min) / (net_max - net_min))
            commercial_components = [value for value in (s_valid, s_moq, s_mpq, s_pay) if value is not None]
            s_com = sum(commercial_components) / len(commercial_components) if commercial_components else None

            if not is_dg:
                s_comp = 1.0
            else:
                s_comp = 0.5 * (1.0 if quote_header.get("un38_3Provided") else 0.0) + 0.5 * (
                    1.0 if quote_header.get("dgPackingSupported") else 0.0
                )

            submitted_at = _parse_iso_datetime(quote_header.get("submittedAtUtc"), line_errors, "submittedAtUtc")
            deadline_utc = _parse_iso_datetime(deadline_value, line_errors, "quoteDeadlineUtc")
            if submitted_at and deadline_utc:
                margin_hours = (deadline_utc - submitted_at).total_seconds() / 3600
                if margin_hours < 0:
                    s_resp = 0.0
                else:
                    s_resp = _clamp(margin_hours / line_constants["resp_ref_hrs"])
            else:
                s_resp = None

            score_soft = _weighted_average(
                {
                    "price": s_price,
                    "lead_time": s_lt,
                    "capacity": s_cap,
                    "site": s_site,
                    "commercial": s_com,
                    "compliance": s_comp,
                    "responsiveness": s_resp,
                },
                line_weights,
            )

            raw_ex = 0.0
            for tag in exception_tags:
                raw_ex += exception_weights.get(tag, 0)
            p_ex = _clamp(raw_ex / line_constants["exception_ref"])
            score_final = _clamp(score_soft - line_constants["exception_weight"] * p_ex)
            line_fit_score = 100 * score_final
            base_line_score, final_line_score = _build_final_line_score(
                line_fit_score,
                supplier_fit_score,
                final_line_config,
            )
            final_line_score *= scenario_score_multiplier
            if exception_tag_set.intersection({"price_adjustment", "price_reopener"}):
                final_line_score *= scenario_exception_score_multiplier
            if is_dg and (
                not quote_header.get("un38_3Provided") or not quote_header.get("dgPackingSupported")
            ):
                final_line_score *= scenario_compliance_score_multiplier
            final_line_score = _clamp(final_line_score, 0.0, 100.0)

            line_scores.append(
                {
                    "rfqLineId": rfq_line_id,
                    "lineNo": line.get("lineNo"),
                    "lineFitScore": line_fit_score,
                    "supplierFitScore": supplier_fit_score,
                    "baseLineScore": base_line_score,
                    "finalLineScore": final_line_score,
                    "subScores": {
                        "price": s_price,
                        "leadTime": s_lt,
                        "capacity": s_cap,
                        "site": s_site,
                        "commercial": s_com,
                        "compliance": s_comp,
                        "responsiveness": s_resp,
                    },
                    "penalties": {
                        "exceptionPenalty": p_ex,
                        "exceptionRaw": raw_ex,
                    },
                    "gates": gates,
                    "inputs": {
                        "unitPrice": unit_price,
                        "leadTimeDays": lt_value,
                        "netDays": net_days,
                        "capacityMonthlyQty": capacity_qty,
                        "requiredMonthlyQty": line_required_monthly_qty,
                        "demandMultiplier": demand_multiplier,
                    },
                    "errors": line_errors,
                }
            )
            if line_errors:
                event_errors.extend([f"{rfq_line_id}: {error}" for error in line_errors])

            event_weighted_total += final_line_score * line_weights_map.get(rfq_line_id, 0)
            event_weight_total += line_weights_map.get(rfq_line_id, 0)

        event_score = event_weighted_total / event_weight_total if event_weight_total else 0.0
        rounded_total = round(event_score, score_decimals)
        score_entry: Dict[str, Any] = {"supplierId": supplier_id, "score": rounded_total}
        if include_details:
            score_entry["breakdown"] = {
                "eventScore": event_score,
                "supplierFitScore": supplier_fit_score,
                "supplierFitBreakdown": supplier_fit_breakdown,
                "lineScores": line_scores,
                "lineWeights": line_weights_map,
                "baseTierName": base_tier_name,
                "requiredMonthlyQty": required_monthly_qty,
                "finalLinePolicy": final_line_config,
                "totalCost": total_cost if total_cost else None,
                "errors": event_errors,
                "scenarioContext": context.get("id"),
            }
        scores.append(score_entry)

    return scores
