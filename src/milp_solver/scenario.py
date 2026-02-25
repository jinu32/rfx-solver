from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from .scoring import score_suppliers


DEFAULT_SCENARIO_PRIORITY = [
    "sourcing_policy",
    "cost_shock",
    "supply_disruption",
    "demand_mix",
    "leadtime_deterioration",
    "quality_compliance",
]

DEFAULT_SCENARIO_LEVEL = "L1"
DEFAULT_LEVELS = ("L0", "L1", "L2")

DEFAULT_LINE_POLICY: Dict[str, Any] = {
    "default": {
        "minSuppliers": 1,
        "maxSuppliers": 2,
        "minShare": 0.0,
        "maxShare": 1.0,
    },
    "perLine": {},
}

SOURCING_POLICY_DEFAULTS: Dict[str, float] = {
    "minSuppliers": 2,
    "maxSuppliers": 2,
    "minShare": 0.30,
    "maxShare": 0.70,
}

DEFAULT_ELIGIBILITY_POLICY: Dict[str, Any] = {
    "enforceSupplierMasterGate": True,
}

DEFAULT_OBJECTIVE_POLICY: Dict[str, Any] = {
    "wCost": 1.0,
    "wScore": 0.02,
    "wRisk": 0.2,
    "normalization": "dynamic",
}

DEFAULT_FALLBACK_POLICY: Dict[str, Any] = {
    "allowSoftFeasible": True,
    "capacitySlackPenalty": 1_000_000.0,
    "assignmentSlackPenalty": 1_000_000.0,
}

SCENARIO_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "demand_mix",
        "name": "Demand/Mix Shift",
        "category": "Demand / Mix",
        "description": "Demand volatility shifts line mix and tier exposure.",
        "levels": {
            "L0": {"demand_multipliers": [1.05, 0.98, 0.97]},
            "L1": {"demand_multipliers": [1.2, 0.9, 0.9]},
            "L2": {"demand_multipliers": [1.35, 0.85, 0.8]},
        },
        "objective": {"wCost": 1.0, "wScore": 0.02, "wRisk": 0.2},
    },
    {
        "id": "supply_disruption",
        "name": "Supply Disruption",
        "category": "Supply disruption",
        "description": "Top supplier capacity shock reduces available output.",
        "levels": {
            "L0": {"capacity_shock_pct": 0.8},
            "L1": {"capacity_shock_pct": 0.65},
            "L2": {"capacity_shock_pct": 0.5},
        },
        "objective": {"wCost": 1.0, "wScore": 0.02, "wRisk": 0.25},
    },
    {
        "id": "leadtime_deterioration",
        "name": "Leadtime Deterioration",
        "category": "Leadtime",
        "description": "Leadtime degradation lowers delivery confidence and effective score.",
        "levels": {
            "L0": {"score_multiplier": 0.97, "cost_multiplier": 1.02},
            "L1": {"score_multiplier": 0.93, "cost_multiplier": 1.05},
            "L2": {"score_multiplier": 0.88, "cost_multiplier": 1.09},
        },
        "objective": {"wCost": 1.0, "wScore": 0.02, "wRisk": 0.22},
    },
    {
        "id": "cost_shock",
        "name": "Cost Shock",
        "category": "Cost shocks",
        "description": "Price adjustment clauses trigger cost escalation.",
        "levels": {
            "L0": {"exception_cost_multiplier": 1.08, "exception_score_multiplier": 0.95},
            "L1": {"exception_cost_multiplier": 1.15, "exception_score_multiplier": 0.9},
            "L2": {"exception_cost_multiplier": 1.25, "exception_score_multiplier": 0.82},
        },
        "objective": {"wCost": 1.0, "wScore": 0.018, "wRisk": 0.22},
    },
    {
        "id": "quality_compliance",
        "name": "Quality/Compliance Incident",
        "category": "Quality & Compliance",
        "description": "Compliance gaps reduce effective line scores.",
        "levels": {
            "L0": {"compliance_score_multiplier": 0.93},
            "L1": {"compliance_score_multiplier": 0.85},
            "L2": {"compliance_score_multiplier": 0.75},
        },
        "objective": {"wCost": 1.0, "wScore": 0.02, "wRisk": 0.24},
    },
    {
        "id": "sourcing_policy",
        "name": "Sourcing Policy Split",
        "category": "Sourcing policy",
        "description": "Dual-source requirement enforces a 70/30 split.",
        "levels": {
            "L0": {},
            "L1": {},
            "L2": {},
        },
        "objective": {"wCost": 1.0, "wScore": 0.02, "wRisk": 0.2},
    },
    {
        "id": "logistics_shift",
        "name": "Logistics Shift",
        "category": "Logistics",
        "description": "SEA to AIR switch adds logistics premium and lead time stress.",
        "levels": {
            "L0": {"cost_multiplier": 1.03, "score_multiplier": 0.98},
            "L1": {"cost_multiplier": 1.08, "score_multiplier": 0.92},
            "L2": {"cost_multiplier": 1.15, "score_multiplier": 0.85},
        },
        "objective": {"wCost": 1.0, "wScore": 0.02, "wRisk": 0.22},
    },
]

SCENARIO_IDS = {template["id"] for template in SCENARIO_TEMPLATES}


try:
    import pulp
except ModuleNotFoundError:  # pragma: no cover - exercised in dedicated fallback test
    pulp = None


def _risk_label(score: float) -> str:
    if score >= 80:
        return "Low"
    if score >= 60:
        return "Medium"
    return "High"


def _sensitivity_level(index: float) -> str:
    if index <= 5:
        return "Low"
    if index <= 15:
        return "Medium"
    return "High"


def _normalize_supplier_status(status: Any) -> str:
    return str(status or "").strip().lower()


def _feasibility_label(feasible: bool, violations: List[Dict[str, Any]]) -> str:
    if not feasible:
        return "Infeasible"
    return "Soft Feasible" if violations else "Feasible"


def _scenario_constraint_tags(scenario: Dict[str, Any], feasibility: str) -> List[str]:
    tags: List[str] = []
    if scenario.get("demand_multipliers"):
        tags.append("demand_mix")
    if scenario.get("capacity_shock_pct"):
        tags.append("capacity_shock")
    if scenario.get("cost_multiplier") or scenario.get("exception_cost_multiplier"):
        tags.append("cost_stress")
    if scenario.get("compliance_score_multiplier"):
        tags.append("compliance_stress")
    if scenario.get("id") == "leadtime_deterioration":
        tags.append("leadtime_deterioration")
    if scenario.get("id") == "sourcing_policy":
        tags.append("line_dual_sourcing")
    if feasibility == "Soft Feasible":
        tags.append("soft_feasible")
    if feasibility == "Infeasible":
        tags.append("infeasible")
    return tags


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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


def _normalize_scenario_level(level: Any) -> str:
    normalized = str(level or "").upper()
    return normalized if normalized in DEFAULT_LEVELS else DEFAULT_SCENARIO_LEVEL


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _sanitize_line_policy_entry(raw: Optional[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        "minSuppliers": _to_int((raw or {}).get("minSuppliers"), _to_int(fallback.get("minSuppliers"), 1)),
        "maxSuppliers": _to_int((raw or {}).get("maxSuppliers"), _to_int(fallback.get("maxSuppliers"), 1)),
        "minShare": _safe_number((raw or {}).get("minShare")),
        "maxShare": _safe_number((raw or {}).get("maxShare")),
    }

    if merged["minShare"] is None:
        merged["minShare"] = float(_safe_number(fallback.get("minShare")) or 0.0)
    if merged["maxShare"] is None:
        merged["maxShare"] = float(_safe_number(fallback.get("maxShare")) or 1.0)

    merged["minSuppliers"] = max(0, merged["minSuppliers"])
    merged["maxSuppliers"] = max(0, merged["maxSuppliers"])

    if merged["maxSuppliers"] < merged["minSuppliers"]:
        merged["maxSuppliers"] = merged["minSuppliers"]

    min_share = _clamp(float(merged["minShare"]), 0.0, 1.0)
    max_share = _clamp(float(merged["maxShare"]), 0.0, 1.0)
    if max_share < min_share:
        max_share = min_share

    merged["minShare"] = min_share
    merged["maxShare"] = max_share
    return merged


def _merge_line_policy(line_policy: Optional[Dict[str, Any]], scenario_id: str) -> Dict[str, Any]:
    base_default = _sanitize_line_policy_entry(DEFAULT_LINE_POLICY.get("default"), DEFAULT_LINE_POLICY["default"])
    merged_default = _sanitize_line_policy_entry((line_policy or {}).get("default"), base_default)

    if scenario_id == "sourcing_policy":
        merged_default = _sanitize_line_policy_entry(SOURCING_POLICY_DEFAULTS, merged_default)

    merged_per_line: Dict[str, Dict[str, Any]] = {}
    raw_per_line = (line_policy or {}).get("perLine") or {}
    if isinstance(raw_per_line, dict):
        for line_id, value in raw_per_line.items():
            if not isinstance(value, dict):
                continue
            merged_per_line[str(line_id)] = _sanitize_line_policy_entry(value, merged_default)

    return {
        "default": merged_default,
        "perLine": merged_per_line,
    }


def _line_policy_for_line(line_id: str, line_policy: Dict[str, Any]) -> Dict[str, Any]:
    per_line = (line_policy.get("perLine") or {}).get(line_id)
    if isinstance(per_line, dict):
        return _sanitize_line_policy_entry(per_line, line_policy.get("default") or DEFAULT_LINE_POLICY["default"])
    return _sanitize_line_policy_entry(line_policy.get("default"), DEFAULT_LINE_POLICY["default"])


def _build_line_inputs_from_scores(
    scores: List[Dict[str, Any]],
    supplier_master: Optional[List[Dict[str, Any]]],
    enforce_supplier_master_gate: bool,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Dict[str, Any]]], List[str]]:
    line_inputs: Dict[str, Dict[str, Dict[str, float]]] = {}
    gate_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    supplier_master_map = {str(entry.get("id")): entry for entry in supplier_master or []}

    for score_entry in scores:
        supplier_id = str(score_entry.get("supplierId") or "")
        if not supplier_id:
            continue

        supplier_gate_reasons: List[str] = []
        supplier_master_entry = supplier_master_map.get(supplier_id)
        if enforce_supplier_master_gate and supplier_master_entry:
            supplier_status = _normalize_supplier_status(supplier_master_entry.get("supplierStatus"))
            if supplier_status in {"blocked", "inactive"}:
                supplier_gate_reasons.append("supplier_master_status")
            if supplier_master_entry.get("msaActiveFlag") is False:
                supplier_gate_reasons.append("supplier_master_msa")

        line_inputs[supplier_id] = {}
        breakdown = score_entry.get("breakdown") or {}
        for line_payload in breakdown.get("lineScores") or []:
            rfq_line_id = str(line_payload.get("rfqLineId") or "")
            if not rfq_line_id:
                continue

            pair_reasons = list(supplier_gate_reasons)
            gates = line_payload.get("gates") or {}
            # DG compliance is a soft scoring factor; only validity/capacity are hard filters for scenario candidates.
            for gate_key in ("validity", "capacity"):
                if gates.get(gate_key) is False:
                    pair_reasons.append(f"line_gate_{gate_key}")

            unit_price = _safe_number((line_payload.get("inputs") or {}).get("unitPrice"))
            if unit_price is None:
                pair_reasons.append("missing_unit_price")

            gate_entry = {
                "status": "pass" if not pair_reasons else "fail",
                "reasons": sorted(set(pair_reasons)),
            }
            gate_map.setdefault(rfq_line_id, {})[supplier_id] = gate_entry

            if pair_reasons:
                continue

            line_inputs[supplier_id][rfq_line_id] = {
                "score": float(_safe_number(line_payload.get("finalLineScore")) or 0.0),
                "unitCost": float(unit_price),
            }

    supplier_ids = sorted(line_inputs.keys())
    return line_inputs, gate_map, supplier_ids


def _scenario_order(scenario_priorities: Optional[List[str]]) -> List[str]:
    if isinstance(scenario_priorities, list) and scenario_priorities:
        selected: List[str] = []
        for scenario_id in scenario_priorities:
            if scenario_id in SCENARIO_IDS and scenario_id not in selected:
                selected.append(scenario_id)
        return selected
    return list(DEFAULT_SCENARIO_PRIORITY)


def _resolve_objective_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = policy or {}
    return {
        "wCost": float(_safe_number(raw.get("wCost")) or DEFAULT_OBJECTIVE_POLICY["wCost"]),
        "wScore": float(_safe_number(raw.get("wScore")) or DEFAULT_OBJECTIVE_POLICY["wScore"]),
        "wRisk": float(_safe_number(raw.get("wRisk")) or DEFAULT_OBJECTIVE_POLICY["wRisk"]),
        "normalization": str(raw.get("normalization") or DEFAULT_OBJECTIVE_POLICY["normalization"]),
    }


def _resolve_fallback_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = policy or {}
    return {
        "allowSoftFeasible": bool(raw.get("allowSoftFeasible", DEFAULT_FALLBACK_POLICY["allowSoftFeasible"])),
        "capacitySlackPenalty": float(
            _safe_number(raw.get("capacitySlackPenalty")) or DEFAULT_FALLBACK_POLICY["capacitySlackPenalty"]
        ),
        "assignmentSlackPenalty": float(
            _safe_number(raw.get("assignmentSlackPenalty")) or DEFAULT_FALLBACK_POLICY["assignmentSlackPenalty"]
        ),
    }


def _resolve_eligibility_policy(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = policy or {}
    return {
        "enforceSupplierMasterGate": bool(
            raw.get(
                "enforceSupplierMasterGate",
                DEFAULT_ELIGIBILITY_POLICY["enforceSupplierMasterGate"],
            )
        ),
    }


def _resolve_risk_by_supplier(
    supplier_ids: List[str],
    supplier_master: Optional[List[Dict[str, Any]]],
) -> Dict[str, float]:
    supplier_master_map = {str(entry.get("id")): entry for entry in supplier_master or []}
    risk_by_supplier: Dict[str, float] = {}
    for supplier_id in supplier_ids:
        supplier_entry = supplier_master_map.get(supplier_id) or {}
        risk_score = _safe_number(supplier_entry.get("riskScore"))
        if risk_score is None:
            risk_by_supplier[supplier_id] = 0.5
        else:
            risk_by_supplier[supplier_id] = _clamp(risk_score / 100.0, 0.0, 1.0)
    return risk_by_supplier


def _build_scenarios_catalog(
    scenario_levels: Optional[Dict[str, str]],
    base_line_policy: Optional[Dict[str, Any]],
    objective_policy: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    level_map = scenario_levels or {}
    by_id: Dict[str, Dict[str, Any]] = {}

    for template in SCENARIO_TEMPLATES:
        scenario_id = template["id"]
        level = _normalize_scenario_level(level_map.get(scenario_id))
        level_payload = (template.get("levels") or {}).get(level) or {}

        merged_objective = {
            **(template.get("objective") or {}),
            **objective_policy,
        }

        scenario_line_policy = _merge_line_policy(base_line_policy, scenario_id)

        by_id[scenario_id] = {
            "id": scenario_id,
            "name": template.get("name"),
            "category": template.get("category"),
            "description": template.get("description"),
            "level": level,
            "objective": merged_objective,
            "linePolicy": scenario_line_policy,
            **level_payload,
        }

    return by_id


def _build_normalization_refs(
    demand_by_line: Dict[str, float],
    supplier_ids: List[str],
    line_inputs: Dict[str, Dict[str, Dict[str, float]]],
    risk_by_supplier: Dict[str, float],
) -> Tuple[float, float, float]:
    cost_ref = 0.0
    score_ref = 0.0
    risk_ref = 0.0

    for line_id, demand in demand_by_line.items():
        unit_costs = [
            float((line_inputs.get(supplier_id, {}).get(line_id) or {}).get("unitCost", 0.0))
            for supplier_id in supplier_ids
            if line_inputs.get(supplier_id, {}).get(line_id)
        ]
        scores = [
            float((line_inputs.get(supplier_id, {}).get(line_id) or {}).get("score", 0.0))
            for supplier_id in supplier_ids
            if line_inputs.get(supplier_id, {}).get(line_id)
        ]
        risks = [
            float(risk_by_supplier.get(supplier_id, 0.5))
            for supplier_id in supplier_ids
            if line_inputs.get(supplier_id, {}).get(line_id)
        ]

        if unit_costs:
            cost_ref += demand * max(unit_costs)
        if scores:
            score_ref += max(scores)
        if risks:
            risk_ref += demand * max(risks)

    return max(cost_ref, 1.0), max(score_ref, 1.0), max(risk_ref, 1.0)


def _capacity_by_supplier(
    response_map: Dict[str, Dict[str, Any]],
    supplier_ids: List[str],
    top_supplier_id: Optional[str],
    capacity_shock_pct: Optional[float],
) -> Dict[str, float]:
    capacity_by_supplier: Dict[str, float] = {}
    for supplier_id in supplier_ids:
        response = response_map.get(supplier_id, {})
        quote_header = response.get("quoteHeader") or {}
        capacity = _safe_number(quote_header.get("capacityMonthlyQty"))
        if capacity is None:
            continue
        if capacity_shock_pct is not None and top_supplier_id and supplier_id == top_supplier_id:
            capacity *= capacity_shock_pct
        capacity_by_supplier[supplier_id] = capacity
    return capacity_by_supplier


def _var_token(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)

def _solve_milp(
    scenario: Dict[str, Any],
    rfq_lines: List[Dict[str, Any]],
    supplier_ids: List[str],
    demand_by_line: Dict[str, float],
    capacity_by_supplier: Dict[str, float],
    line_inputs: Dict[str, Dict[str, Dict[str, float]]],
    gate_map: Dict[str, Dict[str, Dict[str, Any]]],
    risk_by_supplier: Dict[str, float],
    fallback_policy: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    float,
    float,
    bool,
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    if pulp is None:
        return _solve_without_pulp(
            scenario,
            rfq_lines,
            supplier_ids,
            demand_by_line,
            capacity_by_supplier,
            line_inputs,
            gate_map,
            risk_by_supplier,
            fallback_policy,
        )

    line_ids = [str(line.get("rfqLineId") or "") for line in rfq_lines if line.get("rfqLineId")]
    if not line_ids or not supplier_ids:
        return [], 0.0, 0.0, False, [], {"status": "Infeasible", "gapPct": None, "solveMs": 0}, [
            {"type": "infeasible_model", "scope": "global", "message": "No eligible line/supplier candidates."}
        ]

    allow_soft = bool(fallback_policy.get("allowSoftFeasible", True))
    capacity_penalty = float(fallback_policy.get("capacitySlackPenalty", 1_000_000.0))
    assignment_penalty = float(fallback_policy.get("assignmentSlackPenalty", 1_000_000.0))

    model = pulp.LpProblem(f"rfq_scenario_{scenario['id']}", pulp.LpMinimize)
    y = {
        (line_id, supplier_id): pulp.LpVariable(
            f"y_{_var_token(line_id)}_{_var_token(supplier_id)}",
            lowBound=0,
            upBound=1,
            cat="Continuous",
        )
        for line_id in line_ids
        for supplier_id in supplier_ids
    }
    z = {
        (line_id, supplier_id): pulp.LpVariable(
            f"z_{_var_token(line_id)}_{_var_token(supplier_id)}",
            cat="Binary",
        )
        for line_id in line_ids
        for supplier_id in supplier_ids
    }

    slack_assign = {
        line_id: pulp.LpVariable(f"slack_assign_{_var_token(line_id)}", lowBound=0, cat="Continuous")
        for line_id in line_ids
    } if allow_soft else {}

    slack_cap = {
        supplier_id: pulp.LpVariable(f"slack_cap_{_var_token(supplier_id)}", lowBound=0, cat="Continuous")
        for supplier_id in supplier_ids
    } if allow_soft else {}

    objective_policy = scenario.get("objective") or {}
    w_cost = float(_safe_number(objective_policy.get("wCost")) or DEFAULT_OBJECTIVE_POLICY["wCost"])
    w_score = float(_safe_number(objective_policy.get("wScore")) or DEFAULT_OBJECTIVE_POLICY["wScore"])
    w_risk = float(_safe_number(objective_policy.get("wRisk")) or DEFAULT_OBJECTIVE_POLICY["wRisk"])
    normalization_mode = str(objective_policy.get("normalization") or DEFAULT_OBJECTIVE_POLICY["normalization"]).lower()

    cost_term = pulp.lpSum(
        demand_by_line[line_id]
        * line_inputs.get(supplier_id, {}).get(line_id, {}).get("unitCost", 0.0)
        * y[(line_id, supplier_id)]
        for line_id in line_ids
        for supplier_id in supplier_ids
    )

    score_term = pulp.lpSum(
        line_inputs.get(supplier_id, {}).get(line_id, {}).get("score", 0.0)
        * y[(line_id, supplier_id)]
        for line_id in line_ids
        for supplier_id in supplier_ids
    )

    total_demand = sum(float(demand_by_line.get(line_id, 0.0)) for line_id in line_ids)
    risk_term = pulp.lpSum(
        demand_by_line[line_id]
        * float(risk_by_supplier.get(supplier_id, 0.5))
        * y[(line_id, supplier_id)]
        for line_id in line_ids
        for supplier_id in supplier_ids
    ) / max(total_demand, 1.0)

    if normalization_mode == "dynamic":
        cost_ref, score_ref, risk_ref = _build_normalization_refs(
            demand_by_line,
            supplier_ids,
            line_inputs,
            risk_by_supplier,
        )
        objective = w_cost * (cost_term / cost_ref) - w_score * (score_term / score_ref) + w_risk * (risk_term / risk_ref)
    else:
        objective = w_cost * cost_term - w_score * score_term + w_risk * risk_term

    if allow_soft:
        objective += capacity_penalty * pulp.lpSum(slack_cap.values())
        objective += assignment_penalty * pulp.lpSum(slack_assign.values())

    model += objective

    line_policy = scenario.get("linePolicy") or _merge_line_policy(None, scenario.get("id", ""))

    for line_id in line_ids:
        policy_for_line = _line_policy_for_line(line_id, line_policy)
        candidates = [
            supplier_id
            for supplier_id in supplier_ids
            if line_inputs.get(supplier_id, {}).get(line_id)
        ]

        effective_max_suppliers = min(int(policy_for_line["maxSuppliers"]), len(candidates))
        effective_min_suppliers = min(int(policy_for_line["minSuppliers"]), effective_max_suppliers)

        for supplier_id in supplier_ids:
            payload = line_inputs.get(supplier_id, {}).get(line_id)
            if not payload:
                model += y[(line_id, supplier_id)] == 0
                model += z[(line_id, supplier_id)] == 0
                continue

            min_share = float(policy_for_line["minShare"])
            max_share = float(policy_for_line["maxShare"])
            model += y[(line_id, supplier_id)] <= z[(line_id, supplier_id)]
            model += y[(line_id, supplier_id)] >= min_share * z[(line_id, supplier_id)]
            model += y[(line_id, supplier_id)] <= max_share * z[(line_id, supplier_id)]

        if allow_soft:
            model += (
                pulp.lpSum(y[(line_id, supplier_id)] for supplier_id in supplier_ids)
                + slack_assign[line_id]
                == 1,
                f"assign_{_var_token(line_id)}",
            )
        else:
            model += (
                pulp.lpSum(y[(line_id, supplier_id)] for supplier_id in supplier_ids)
                == 1,
                f"assign_{_var_token(line_id)}",
            )

        model += (
            pulp.lpSum(z[(line_id, supplier_id)] for supplier_id in supplier_ids) >= effective_min_suppliers,
            f"min_suppliers_{_var_token(line_id)}",
        )
        model += (
            pulp.lpSum(z[(line_id, supplier_id)] for supplier_id in supplier_ids) <= effective_max_suppliers,
            f"max_suppliers_{_var_token(line_id)}",
        )

    for supplier_id in supplier_ids:
        capacity = capacity_by_supplier.get(supplier_id)
        if capacity is None:
            continue
        capacity_expr = pulp.lpSum(
            demand_by_line[line_id] * y[(line_id, supplier_id)]
            for line_id in line_ids
        )
        if allow_soft:
            model += capacity_expr <= capacity + slack_cap[supplier_id], f"capacity_{_var_token(supplier_id)}"
        else:
            model += capacity_expr <= capacity, f"capacity_{_var_token(supplier_id)}"

    start_ts = perf_counter()
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)
    solve_ms = int((perf_counter() - start_ts) * 1000)

    status = pulp.LpStatus.get(model.status, str(model.status))
    solver_payload = {
        "status": status,
        "gapPct": None,
        "solveMs": solve_ms,
    }

    feasible = status not in {"Infeasible", "Unbounded", "Undefined"}
    if not feasible:
        line_summary = [
            {
                "rfqLineId": line_id,
                "assignedSupplierCount": 0,
                "unassignedShare": 1.0,
            }
            for line_id in line_ids
        ]
        return (
            [],
            0.0,
            0.0,
            False,
            line_summary,
            solver_payload,
            [{"type": "infeasible_model", "scope": "global", "message": "MILP model is infeasible."}],
        )

    violations: List[Dict[str, Any]] = []
    award_plan: List[Dict[str, Any]] = []
    line_summary: List[Dict[str, Any]] = []
    computed_cost = 0.0
    score_accumulator = 0.0

    for line_id in line_ids:
        assigned_count = 0
        assigned_share = 0.0
        for supplier_id in supplier_ids:
            share_value = float(pulp.value(y[(line_id, supplier_id)]) or 0.0)
            if share_value <= 1e-9:
                continue
            payload = line_inputs.get(supplier_id, {}).get(line_id)
            if not payload:
                continue
            assigned_count += 1
            assigned_share += share_value

            unit_cost = float(payload.get("unitCost", 0.0))
            raw_line_score = float(payload.get("score", 0.0))
            line_tco = demand_by_line[line_id] * unit_cost * share_value
            score_accumulator += raw_line_score * share_value
            computed_cost += line_tco

            gate_entry = (gate_map.get(line_id) or {}).get(supplier_id) or {"status": "pass", "reasons": []}
            award_plan.append(
                {
                    "rfqLineId": line_id,
                    "supplierId": supplier_id,
                    "selected": bool(float(pulp.value(z[(line_id, supplier_id)]) or 0.0) >= 0.5),
                    "allocationPct": share_value,
                    "awardShare": share_value,
                    "award_share": share_value,
                    "awardQty": demand_by_line[line_id] * share_value,
                    "award_qty": demand_by_line[line_id] * share_value,
                    "lineScore": raw_line_score,
                    "lineCost": line_tco,
                    "lineTco": line_tco,
                    "lineTCO": line_tco,
                    "gateStatus": gate_entry.get("status", "pass"),
                    "gateReasons": gate_entry.get("reasons", []),
                }
            )

        unassigned_share = max(0.0, 1.0 - assigned_share)
        if allow_soft:
            unassigned_share = float(pulp.value(slack_assign[line_id]) or 0.0)
        if unassigned_share > 1e-6:
            violations.append(
                {
                    "type": "unassigned_share",
                    "scope": "line",
                    "rfqLineId": line_id,
                    "value": round(unassigned_share, 6),
                    "message": f"Line {line_id} has {unassigned_share:.4f} unassigned share.",
                }
            )

        line_summary.append(
            {
                "rfqLineId": line_id,
                "assignedSupplierCount": assigned_count,
                "unassignedShare": unassigned_share,
            }
        )

    if allow_soft:
        for supplier_id in supplier_ids:
            cap_slack = float(pulp.value(slack_cap[supplier_id]) or 0.0)
            if cap_slack <= 1e-6:
                continue
            violations.append(
                {
                    "type": "capacity_overflow",
                    "scope": "supplier",
                    "supplierId": supplier_id,
                    "value": round(cap_slack, 6),
                    "message": f"Supplier {supplier_id} exceeded capacity by {cap_slack:.4f}.",
                }
            )

    total_score = score_accumulator / max(len(line_ids), 1)
    return award_plan, computed_cost, total_score, True, line_summary, solver_payload, violations


def _solve_without_pulp(
    scenario: Dict[str, Any],
    rfq_lines: List[Dict[str, Any]],
    supplier_ids: List[str],
    demand_by_line: Dict[str, float],
    capacity_by_supplier: Dict[str, float],
    line_inputs: Dict[str, Dict[str, Dict[str, float]]],
    gate_map: Dict[str, Dict[str, Dict[str, Any]]],
    risk_by_supplier: Dict[str, float],
    fallback_policy: Dict[str, Any],
) -> Tuple[
    List[Dict[str, Any]],
    float,
    float,
    bool,
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    allow_soft = bool(fallback_policy.get("allowSoftFeasible", True))
    line_policy = scenario.get("linePolicy") or _merge_line_policy(None, scenario.get("id", ""))

    objective_policy = scenario.get("objective") or {}
    w_cost = float(_safe_number(objective_policy.get("wCost")) or DEFAULT_OBJECTIVE_POLICY["wCost"])
    w_score = float(_safe_number(objective_policy.get("wScore")) or DEFAULT_OBJECTIVE_POLICY["wScore"])
    w_risk = float(_safe_number(objective_policy.get("wRisk")) or DEFAULT_OBJECTIVE_POLICY["wRisk"])

    remaining_capacity = dict(capacity_by_supplier)
    award_plan: List[Dict[str, Any]] = []
    line_summary: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    total_cost = 0.0
    score_accumulator = 0.0

    for line in rfq_lines:
        line_id = str(line.get("rfqLineId") or "")
        if not line_id:
            continue

        demand = float(demand_by_line.get(line_id, 0.0))
        policy_for_line = _line_policy_for_line(line_id, line_policy)

        candidates: List[Tuple[float, str, Dict[str, float]]] = []
        for supplier_id in supplier_ids:
            payload = line_inputs.get(supplier_id, {}).get(line_id)
            if not payload:
                continue
            unit_cost = float(payload.get("unitCost", 0.0))
            raw_score = float(payload.get("score", 0.0))
            risk = float(risk_by_supplier.get(supplier_id, 0.5))
            objective_value = (w_cost * unit_cost) - (w_score * raw_score) + (w_risk * risk)
            candidates.append((objective_value, supplier_id, payload))

        candidates.sort(key=lambda item: item[0])

        if not candidates:
            if allow_soft:
                line_summary.append(
                    {
                        "rfqLineId": line_id,
                        "assignedSupplierCount": 0,
                        "unassignedShare": 1.0,
                    }
                )
                violations.append(
                    {
                        "type": "unassigned_share",
                        "scope": "line",
                        "rfqLineId": line_id,
                        "value": 1.0,
                        "message": f"Line {line_id} has no eligible supplier candidates.",
                    }
                )
                continue
            return (
                [],
                0.0,
                0.0,
                False,
                [],
                {"status": "Infeasible", "gapPct": None, "solveMs": 0},
                [{"type": "infeasible_model", "scope": "line", "rfqLineId": line_id, "message": "No candidates"}],
            )

        target_supplier_count = max(1, int(policy_for_line["minSuppliers"]))
        target_supplier_count = min(target_supplier_count, int(policy_for_line["maxSuppliers"]), len(candidates))

        chosen = candidates[:target_supplier_count]
        remaining_share = 1.0
        assigned_count = 0

        for index, (_, supplier_id, payload) in enumerate(chosen):
            min_share = float(policy_for_line["minShare"])
            max_share = float(policy_for_line["maxShare"])

            if target_supplier_count == 1:
                target_share = 1.0
            elif target_supplier_count == 2:
                if index == 0:
                    target_share = _clamp(1.0 - min_share, min_share, max_share)
                else:
                    target_share = remaining_share
            else:
                target_share = remaining_share / max(target_supplier_count - index, 1)
                target_share = _clamp(target_share, min_share, max_share)

            available_share = target_share
            capacity = remaining_capacity.get(supplier_id)
            if capacity is not None and demand > 0:
                available_share = min(available_share, max(0.0, capacity / demand))

            if available_share <= 1e-9:
                continue

            share = min(available_share, remaining_share)
            if share <= 1e-9:
                continue

            remaining_share -= share
            assigned_count += 1

            if capacity is not None:
                remaining_capacity[supplier_id] = max(0.0, capacity - demand * share)

            line_tco = demand * float(payload.get("unitCost", 0.0)) * share
            line_score = float(payload.get("score", 0.0))
            total_cost += line_tco
            score_accumulator += line_score * share

            gate_entry = (gate_map.get(line_id) or {}).get(supplier_id) or {"status": "pass", "reasons": []}
            award_plan.append(
                {
                    "rfqLineId": line_id,
                    "supplierId": supplier_id,
                    "selected": True,
                    "allocationPct": share,
                    "awardShare": share,
                    "award_share": share,
                    "awardQty": demand * share,
                    "award_qty": demand * share,
                    "lineScore": line_score,
                    "lineCost": line_tco,
                    "lineTco": line_tco,
                    "lineTCO": line_tco,
                    "gateStatus": gate_entry.get("status", "pass"),
                    "gateReasons": gate_entry.get("reasons", []),
                }
            )

        if remaining_share > 1e-9:
            if allow_soft:
                violations.append(
                    {
                        "type": "unassigned_share",
                        "scope": "line",
                        "rfqLineId": line_id,
                        "value": round(remaining_share, 6),
                        "message": f"Line {line_id} has {remaining_share:.4f} unassigned share.",
                    }
                )
            else:
                return (
                    [],
                    0.0,
                    0.0,
                    False,
                    [],
                    {"status": "Infeasible", "gapPct": None, "solveMs": 0},
                    [
                        {
                            "type": "infeasible_model",
                            "scope": "line",
                            "rfqLineId": line_id,
                            "message": "Unable to satisfy full line demand without slack.",
                        }
                    ],
                )

        line_summary.append(
            {
                "rfqLineId": line_id,
                "assignedSupplierCount": assigned_count,
                "unassignedShare": max(0.0, remaining_share),
            }
        )

    total_score = score_accumulator / max(len(rfq_lines), 1)
    solver_payload = {
        "status": "HeuristicFallback",
        "gapPct": None,
        "solveMs": 0,
    }
    return award_plan, total_cost, total_score, True, line_summary, solver_payload, violations


def _build_sensitivity_payload(
    feasible: bool,
    total_cost: float,
    total_score: float,
    baseline_cost: Optional[float],
    baseline_score: Optional[float],
) -> Dict[str, Any]:
    if not feasible:
        return {
            "level": "High",
            "index": None,
            "costDeltaPct": None,
            "scoreDeltaPct": None,
        }

    if baseline_cost is None or baseline_cost <= 0 or baseline_score is None or baseline_score <= 0:
        return {
            "level": "Unknown",
            "index": None,
            "costDeltaPct": None,
            "scoreDeltaPct": None,
        }

    cost_delta_pct = ((total_cost - baseline_cost) / baseline_cost) * 100
    score_delta_pct = ((total_score - baseline_score) / baseline_score) * 100
    sensitivity_index = abs(cost_delta_pct) + abs(score_delta_pct)
    return {
        "level": _sensitivity_level(sensitivity_index),
        "index": sensitivity_index,
        "costDeltaPct": cost_delta_pct,
        "scoreDeltaPct": score_delta_pct,
    }


def build_scenarios(
    rfq: Dict[str, Any],
    response_details: List[Dict[str, Any]],
    scores: List[Dict[str, Any]],
    supplier_master: Optional[List[Dict[str, Any]]] = None,
    scenario_priorities: Optional[List[str]] = None,
    scenario_levels: Optional[Dict[str, str]] = None,
    line_policy: Optional[Dict[str, Any]] = None,
    eligibility_policy: Optional[Dict[str, Any]] = None,
    objective_policy: Optional[Dict[str, Any]] = None,
    fallback_policy: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    print("############## rfx-solver build_scenarios is called ###################")
    rfq_detail = rfq.get("detail") or {}
    rfq_lines = rfq_detail.get("rfqLines") or []
    if not rfq_lines:
        return []

    rfq_volume_tiers = rfq_detail.get("rfqVolumeTiers") or []
    _, base_tier_qty = _select_base_tier(rfq_volume_tiers)
    base_demand_qty = float(base_tier_qty or 1.0)

    response_map = {str(entry.get("supplierId")): entry for entry in response_details if entry.get("supplierId")}
    supplier_name_map = {str(supplier.get("id")): supplier.get("name") for supplier in rfq.get("suppliers") or []}

    top_supplier_id = None
    if scores:
        top_supplier_id = str(max(scores, key=lambda entry: entry.get("score", 0)).get("supplierId") or "") or None

    resolved_objective_policy = _resolve_objective_policy(objective_policy)
    resolved_fallback_policy = _resolve_fallback_policy(fallback_policy)
    resolved_eligibility_policy = _resolve_eligibility_policy(eligibility_policy)
    enforce_supplier_master_gate = bool(resolved_eligibility_policy.get("enforceSupplierMasterGate", True))

    scenario_catalog = _build_scenarios_catalog(
        scenario_levels,
        line_policy,
        resolved_objective_policy,
    )

    selected_ids = _scenario_order(scenario_priorities)
    selected_scenarios = [scenario_catalog[scenario_id] for scenario_id in selected_ids if scenario_id in scenario_catalog]
    if not selected_scenarios:
        return []

    priority_rank = {scenario_id: idx + 1 for idx, scenario_id in enumerate(selected_ids)}

    baseline_demand_by_line = {
        str(line.get("rfqLineId")): base_demand_qty
        for line in rfq_lines
        if line.get("rfqLineId")
    }

    baseline_line_inputs, baseline_gate_map, baseline_supplier_ids = _build_line_inputs_from_scores(
        scores,
        supplier_master,
        enforce_supplier_master_gate,
    )
    baseline_capacity_by_supplier = _capacity_by_supplier(
        response_map,
        baseline_supplier_ids,
        top_supplier_id=None,
        capacity_shock_pct=None,
    )
    baseline_risk_map = _resolve_risk_by_supplier(baseline_supplier_ids, supplier_master)

    baseline_reference_scenario = {
        "id": "base_case_reference",
        "objective": resolved_objective_policy,
        "linePolicy": _merge_line_policy(line_policy, "base_case_reference"),
    }

    (
        _,
        baseline_total_cost,
        baseline_total_score,
        baseline_feasible,
        _,
        _,
        baseline_violations,
    ) = _solve_milp(
        baseline_reference_scenario,
        rfq_lines,
        baseline_supplier_ids,
        baseline_demand_by_line,
        baseline_capacity_by_supplier,
        baseline_line_inputs,
        baseline_gate_map,
        baseline_risk_map,
        resolved_fallback_policy,
    )

    baseline_cost_value = baseline_total_cost if baseline_feasible and not baseline_violations else None
    baseline_score_value = baseline_total_score if baseline_feasible and not baseline_violations else None

    scenario_results: List[Dict[str, Any]] = []

    for scenario in selected_scenarios:
        demand_by_line: Dict[str, float] = {}
        multipliers = scenario.get("demand_multipliers") or []
        for idx, line in enumerate(rfq_lines):
            line_id = str(line.get("rfqLineId") or "")
            if not line_id:
                continue
            multiplier = 1.0
            if multipliers:
                multiplier = float(_safe_number(multipliers[idx % len(multipliers)]) or 1.0)
            demand_by_line[line_id] = base_demand_qty * multiplier

        capacity_shock_pct = _safe_number(scenario.get("capacity_shock_pct"))
        capacity_by_supplier = _capacity_by_supplier(
            response_map,
            [str(entry.get("supplierId") or "") for entry in scores if entry.get("supplierId")],
            top_supplier_id=top_supplier_id,
            capacity_shock_pct=capacity_shock_pct,
        )

        scenario_context = {
            "id": scenario.get("id"),
            "demand_multipliers": scenario.get("demand_multipliers"),
            "capacity_shock_pct": capacity_shock_pct,
            "capacity_shock_supplier_id": top_supplier_id,
            "cost_multiplier": scenario.get("cost_multiplier"),
            "score_multiplier": scenario.get("score_multiplier"),
            "exception_cost_multiplier": scenario.get("exception_cost_multiplier"),
            "exception_score_multiplier": scenario.get("exception_score_multiplier"),
            "compliance_score_multiplier": scenario.get("compliance_score_multiplier"),
        }

        scenario_scores = score_suppliers(
            rfq,
            response_details,
            supplier_master=supplier_master,
            include_details=True,
            scenario_context=scenario_context,
        )

        adjusted_line_inputs, gate_map, scenario_supplier_ids = _build_line_inputs_from_scores(
            scenario_scores,
            supplier_master,
            enforce_supplier_master_gate,
        )

        scenario_capacity_by_supplier = {
            supplier_id: capacity
            for supplier_id, capacity in capacity_by_supplier.items()
            if supplier_id in scenario_supplier_ids
        }
        risk_by_supplier = _resolve_risk_by_supplier(scenario_supplier_ids, supplier_master)

        (
            award_plan,
            total_cost,
            total_score,
            feasible,
            line_summary,
            solver_payload,
            violations,
        ) = _solve_milp(
            scenario,
            rfq_lines,
            scenario_supplier_ids,
            demand_by_line,
            scenario_capacity_by_supplier,
            adjusted_line_inputs,
            gate_map,
            risk_by_supplier,
            resolved_fallback_policy,
        )

        scenario_feasibility = _feasibility_label(feasible, violations)
        scenario_constraint_tags = _scenario_constraint_tags(scenario, scenario_feasibility)

        for entry in award_plan:
            supplier_id = str(entry.get("supplierId") or "")
            line_id = str(entry.get("rfqLineId") or "")
            entry["supplierName"] = supplier_name_map.get(supplier_id, supplier_id)
            entry["awardShare"] = float(entry.get("awardShare", entry.get("allocationPct", 0.0)) or 0.0)
            entry["allocationPct"] = entry["awardShare"]
            demand_qty = float(demand_by_line.get(line_id, 0.0) or 0.0)
            entry["awardQty"] = demand_qty * float(entry.get("awardShare", 0.0) or 0.0)
            entry["award_qty"] = entry["awardQty"]
            entry["lineTco"] = float(entry.get("lineTco", entry.get("lineCost", 0.0)) or 0.0)
            entry["lineTCO"] = entry["lineTco"]
            entry["lineCost"] = entry["lineTco"]
            entry["feasibility"] = scenario_feasibility
            entry["constraintTags"] = list(scenario_constraint_tags)

            gate_entry = (gate_map.get(line_id) or {}).get(supplier_id)
            if gate_entry:
                entry["gateStatus"] = gate_entry.get("status", "pass")
                entry["gateReasons"] = gate_entry.get("reasons", [])
            else:
                entry["gateStatus"] = "pass"
                entry["gateReasons"] = []

        for item in line_summary:
            item["policy"] = _line_policy_for_line(str(item.get("rfqLineId") or ""), scenario.get("linePolicy") or {})

        risk_level = "High"
        if scenario_feasibility == "Feasible":
            risk_level = _risk_label(total_score)
        elif scenario_feasibility == "Soft Feasible":
            risk_level = "High"

        scenario_results.append(
            {
                "id": scenario["id"],
                "priority": priority_rank.get(scenario["id"], len(priority_rank) + 1),
                "level": scenario.get("level"),
                "name": scenario["name"],
                "category": scenario["category"],
                "description": scenario["description"],
                "awardPlan": award_plan,
                "lineSummary": line_summary,
                "solver": solver_payload,
                "violations": violations,
                "tco": total_cost,
                "totalCost": total_cost,
                "totalScore": total_score,
                "feasibility": scenario_feasibility,
                "riskLevel": risk_level,
                "constraintTags": scenario_constraint_tags,
                "sensitivity": _build_sensitivity_payload(
                    feasible and scenario_feasibility == "Feasible",
                    total_cost,
                    total_score,
                    baseline_cost_value,
                    baseline_score_value,
                ),
            }
        )

    return sorted(scenario_results, key=lambda item: item.get("priority", 999))
