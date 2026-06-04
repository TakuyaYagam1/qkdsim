#!/usr/bin/env python3
"""Автоподбор углов для ViPNet QKDSim (практикум по КРК).

Схема: лазер (горизонтальная поляризация) -> λ/2 (θ1) -> λ/4 (θ2)  [Алиса]
       -> λ/2 (θ3) -> λ/4 (θ4) -> куб PBS -> PDH (H) и PDV (V)       [Боб]
Полная мощность в симуляторе: PDH + PDV = 55639.

Команды:
  check  θ1 θ2 θ3 θ4            уровни PDH и PDV для заданных углов
  alice  --bob θ3 θ4 [--target h0|v0] [--theta2 X]
                                подбор θ1, θ2 Алисы: на выбранном детекторе 0
  bob    --alice θ1 θ2 [--target h0|v0] [--theta4 X]
                                подбор θ3, θ4 Боба: на выбранном детекторе 0
  bb84   [--b0 θ3 θ4] [--th3 θ3'] [--root N]
                                таблица правил BB84 (два базиса, четыре состояния)
  goks   [--b00 θ3 θ4] [--sol N] [--flip]
                                таблица правил ГОКС (практикум №6, только PDH)
  b92    [--b0 θ3 θ4] [--flip]  таблица правил B92 (практикум №7, только PDH)
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np

FULL = 55639.0


def _finite_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Угол должен быть числом.") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("Угол должен быть конечным числом.")
    return result


def _require_finite_angles(*angles: float | np.ndarray) -> None:
    if not all(bool(np.all(np.isfinite(angle))) for angle in angles):
        raise ValueError("Углы должны быть конечными числами.")


# ---------------------------------------------------------------- модель стенда
def _hwp(x, y, t):
    c, s = np.cos(2 * np.radians(t)), np.sin(2 * np.radians(t))
    return c * x + s * y, s * x - c * y


def _qwp(x, y, t):
    c, s = np.cos(np.radians(t)), np.sin(np.radians(t))
    a, b = c * x + s * y, 1j * (-s * x + c * y)
    return c * a - s * b, s * a + c * b


def jones(t1, t2, t3, t4):
    """Амплитуды (x, y) на входе PDH и PDV; углы могут быть числами или массивами numpy."""
    _require_finite_angles(t1, t2, t3, t4)
    x = np.ones(np.broadcast(t1, t2, t3, t4).shape, dtype=complex)
    y = np.zeros_like(x)
    for f, t in ((_hwp, t1), (_qwp, t2), (_hwp, t3), (_qwp, t4)):
        x, y = f(x, y, t)
    return x, y


def levels(t1, t2, t3, t4):
    """Уровни сигнала (PDH, PDV)."""
    x, y = jones(t1, t2, t3, t4)
    return FULL * abs(x) ** 2, FULL * abs(y) ** 2


# ---------------------------------------------------------------- решатель
# role == "alice": ищем (θ1, θ2) при заданных углах Боба;
# role == "bob":   ищем (θ3, θ4) при заданных углах Алисы.
def _amp(role, fixed, target, p1, p2):
    x, y = jones(p1, p2, *fixed) if role == "alice" else jones(*fixed, p1, p2)
    return x if target == "h0" else y


def _cost(role, fixed, target, p1, p2):
    return FULL * abs(_amp(role, fixed, target, p1, p2)) ** 2


def _residual(role, fixed, target, p):
    a = _amp(role, fixed, target, p[0], p[1])
    return np.array([float(a.real), float(a.imag)])


def _refine(
    role: str,
    fixed: tuple[float, float],
    target: str,
    seed: tuple[float, float],
    free: list[int],
) -> np.ndarray:
    """Refine a grid seed without rounding the angle or amplitude residual."""
    p = np.array(seed, dtype=float)
    lam = 1e-6
    r = _residual(role, fixed, target, p)
    for _ in range(150):
        if float(np.linalg.norm(r)) <= 1e-14:
            break
        J = np.zeros((2, len(free)))
        for k, i in enumerate(free):
            d = np.zeros(2)
            d[i] = 1e-4
            J[:, k] = (
                _residual(role, fixed, target, p + d)
                - _residual(role, fixed, target, p - d)
            ) / 2e-4
        normal = J.T @ J
        gradient = J.T @ r
        try:
            step = np.linalg.solve(normal + lam * np.eye(len(free)), -gradient)
        except np.linalg.LinAlgError:
            break
        step_norm = float(np.linalg.norm(step))
        if step_norm > 5.0:
            step *= 5.0 / step_norm
        q = p.copy()
        q[free] += step
        rq = _residual(role, fixed, target, q)
        if rq @ rq < r @ r:
            p, r, lam = q, rq, lam / 10
            if float(np.max(np.abs(step))) < 1e-13:
                break
        else:
            lam *= 10
            if lam > 1e12:
                break
    return p


def _periodic_distance(first: float, second: float, period: float) -> float:
    delta = abs((first - second) % period)
    return min(delta, period - delta)


def solve(
    fixed: tuple[float, float],
    target: str,
    second: float | None = None,
    role: str = "alice",
) -> list[tuple[float, float]]:
    """Пары (первый, второй) угол пластин λ/2 и λ/4, при которых выбранный детектор получает 0.

    fixed  - углы другой стороны (для Алисы это (θ3, θ4) Боба, для Боба - (θ1, θ2) Алисы);
    target - 'h0': PDH = 0, 'v0': PDV = 0;
    second - если задан, угол λ/4-пластины фиксируется и ищется только угол λ/2-пластины.
    Результаты возвращаются в полной точности и сортируются по углу λ/4.
    """
    if role not in {"alice", "bob"}:
        raise ValueError("role должен быть alice или bob")
    if target not in {"h0", "v0"}:
        raise ValueError("target должен быть h0 или v0")
    _require_finite_angles(*fixed, *((second,) if second is not None else ()))

    g1 = np.arange(0, 90, 0.5)
    if second is None:
        g2 = np.arange(0, 180, 0.5)
        C = _cost(role, fixed, target, g1[:, None], g2[None, :])
        n1, n2 = C.shape
        seeds = [
            (g1[i], g2[j])
            for i in range(n1)
            for j in range(n2)
            if C[i, j] < 3000
            and C[i, j]
            <= min(
                C[(i + di) % n1, (j + dj) % n2]
                for di in (-1, 0, 1)
                for dj in (-1, 0, 1)
            )
        ]
        free = [0, 1]
        zero_grid_indices = np.argwhere(C <= FULL * 1e-24)
        if len(zero_grid_indices) > 16:
            i, j = zero_grid_indices[0]
            seeds = [(float(g1[i]), float(g2[j]))]
    else:
        C = _cost(role, fixed, target, g1, second)
        n1 = len(g1)
        seeds = [
            (g1[i], second)
            for i in range(n1)
            if C[i] < 3000 and C[i] <= min(C[(i - 1) % n1], C[(i + 1) % n1])
        ]
        free = [0]
        zero_line_indices = np.flatnonzero(C <= FULL * 1e-24)
        if len(zero_line_indices) > 16:
            i = int(zero_line_indices[0])
            seeds = [(float(g1[i]), float(second))]

    out: list[tuple[float, float]] = []
    for seed in seeds:
        if float(np.linalg.norm(_residual(role, fixed, target, seed))) <= 1e-14:
            p = np.array(seed, dtype=float)
        else:
            p = _refine(role, fixed, target, seed, free)
        t1 = float(p[0] % 90)
        t2 = float(second) if second is not None else float(p[1] % 180)
        residual = float(abs(_amp(role, fixed, target, t1, t2)))
        if residual > 1e-12:
            continue
        if any(
            _periodic_distance(t1, old[0], 90) <= 1e-7
            and _periodic_distance(t2, old[1], 180) <= 1e-7
            for old in out
        ):
            continue
        out.append((t1, t2))
    return sorted(out, key=lambda q: (q[1], q[0]))


def _bisect_root(function, left: float, right: float) -> float:
    f_left = function(left)
    f_right = function(right)
    if abs(f_left) <= 1e-14:
        return left
    if abs(f_right) <= 1e-14:
        return right
    for _ in range(100):
        middle = (left + right) / 2
        f_middle = function(middle)
        if abs(f_middle) <= 1e-14 or right - left <= 1e-13:
            return middle
        if f_left * f_middle < 0:
            right, f_right = middle, f_middle
        else:
            left, f_left = middle, f_middle
    return (left + right) / 2


def _minimize_abs(function, left: float, right: float) -> float:
    ratio = (math.sqrt(5) - 1) / 2
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    first_value = function(first) ** 2
    second_value = function(second) ** 2
    for _ in range(100):
        if right - left <= 1e-13:
            break
        if first_value <= second_value:
            right, second, second_value = second, first, first_value
            first = right - ratio * (right - left)
            first_value = function(first) ** 2
        else:
            left, first, first_value = first, second, second_value
            second = left + ratio * (right - left)
            second_value = function(second) ** 2
    return (left + right) / 2


def theta4_equal_split(alice: tuple[float, float], theta3: float) -> list[float]:
    """Return full-precision theta4 roots where both detectors receive half power."""
    d = np.arange(0, 180, 0.005)
    h, v = levels(alice[0], alice[1], theta3, d)
    diff = np.asarray((h - v) / FULL, dtype=float)
    tolerance = 1e-10
    if np.all(np.abs(diff) <= 1e-14):
        return [0.0]

    def residual(theta4: float) -> float:
        h_value, v_value = levels(alice[0], alice[1], theta3, theta4 % 180)
        return float((h_value - v_value) / FULL)

    candidates = [float(d[i]) for i in np.flatnonzero(np.abs(diff) <= 1e-14)]
    for i in range(len(d) - 1):
        if diff[i] * diff[i + 1] < 0:
            candidates.append(_bisect_root(residual, float(d[i]), float(d[i + 1])))

    if diff[-1] * diff[0] < 0:
        candidates.append(_bisect_root(residual, float(d[-1]), 180.0))

    for i in range(1, len(d) - 1):
        if abs(diff[i]) <= abs(diff[i - 1]) and abs(diff[i]) <= abs(diff[i + 1]):
            candidate = _minimize_abs(residual, float(d[i - 1]), float(d[i + 1]))
            if abs(residual(candidate)) <= tolerance:
                candidates.append(candidate)

    roots: list[float] = []
    for candidate in candidates:
        root = candidate % 180
        if abs(residual(root)) > tolerance:
            continue
        if any(_periodic_distance(root, old, 180) <= 1e-6 for old in roots):
            continue
        roots.append(root)
    return sorted(roots)


def fmt(x: float, digits: int = 6) -> str:
    value = _rounded(x, digits)
    if value == 0:
        return "0"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _rounded(value: float, digits: int) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0 else rounded


def shift90(t, delta):
    """Shift theta1 modulo 90 degrees without display rounding."""
    return (t + delta) % 90


ProtocolKey = tuple[int, int]
AnglePair = tuple[float, float]
Decoder = tuple[int | str, int | str]


class ProtocolRow(TypedDict):
    basis: int
    bit: int
    theta1: float
    theta2: float
    theta3: float
    theta4: float
    pdh: int | str
    pdv: int | str


class ProtocolResult(TypedDict):
    protocol: str
    rows: list[ProtocolRow]
    pdh_matrix: list[list[float]]
    max_error: float
    digits: int


IDEAL_PDH: dict[str, list[list[float]]] = {
    "bb84": [
        [0.0, 0.0, 0.5, 0.5],
        [1.0, 1.0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
    ],
    "goks": [
        [0.5, 0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0, 0.5],
        [0.5, 1.0, 0.5, 0.0],
        [1.0, 0.5, 0.0, 0.5],
    ],
    "b92": [[0.5, 0.0], [0.0, 0.5]],
}


def _first_solution(solutions: list[AnglePair], description: str) -> AnglePair:
    if not solutions:
        sys.exit(f"Решений нет: {description}.")
    return solutions[0]


def _protocol_payload(
    protocol: str,
    keys: list[ProtocolKey],
    alice: dict[ProtocolKey, AnglePair],
    bob: dict[ProtocolKey, AnglePair],
    decoder: dict[ProtocolKey, Decoder],
    digits: int,
) -> ProtocolResult:
    rows: list[ProtocolRow] = []
    rendered: dict[ProtocolKey, tuple[AnglePair, AnglePair]] = {}
    for key in keys:
        t1, t2 = (_rounded(value, digits) for value in alice[key])
        t3, t4 = (_rounded(value, digits) for value in bob[key])
        sender = (t1, t2)
        receiver = (t3, t4)
        rendered[key] = sender, receiver
        rows.append(
            {
                "basis": key[0],
                "bit": key[1],
                "theta1": t1,
                "theta2": t2,
                "theta3": t3,
                "theta4": t4,
                "pdh": decoder[key][0],
                "pdv": decoder[key][1],
            }
        )

    matrix: list[list[float]] = []
    for row_key in keys:
        sender, _ = rendered[row_key]
        values = []
        for column_key in keys:
            _, receiver = rendered[column_key]
            pdh = float(levels(*sender, *receiver)[0]) / FULL
            values.append(min(1.0, max(0.0, pdh)))
        matrix.append(values)
    max_error = max(
        abs(matrix[i][j] - IDEAL_PDH[protocol][i][j])
        for i in range(len(keys))
        for j in range(len(keys))
    )
    return {
        "protocol": protocol,
        "rows": rows,
        "pdh_matrix": matrix,
        "max_error": max_error,
        "digits": digits,
    }


def _emit_protocol(result: ProtocolResult, json_output: bool, digits: int) -> None:
    if json_output:
        print(
            json.dumps(
                result, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        )
        return

    protocol = str(result["protocol"])
    title = {"bb84": "BB84", "goks": "ГОКС", "b92": "B92"}[protocol]
    rows = result["rows"]
    matrix = result["pdh_matrix"]
    print(f"Протокол {title}; углы округлены до {digits} знаков")
    print("Базис Бит theta1 theta2 theta3 theta4 PDH PDV")
    for row in rows:
        print(
            f"{row['basis']:>5} {row['bit']:>3} {fmt(row['theta1'], digits):>12}"
            f" {fmt(row['theta2'], digits):>12} {fmt(row['theta3'], digits):>12}"
            f" {fmt(row['theta4'], digits):>12} {row['pdh']!s:>3} {row['pdv']!s:>3}"
        )
    labels = [f"({row['basis']},{row['bit']})" for row in rows]
    print("Матрица PDH в долях; порядок строк и столбцов: " + ", ".join(labels))
    print("      " + " ".join(f"{label:>12}" for label in labels))
    for label, values in zip(labels, matrix):
        print(
            f"{label:>6} " + " ".join(f"{fmt(value, digits):>12}" for value in values)
        )
    print(f"max_error={float(result['max_error']):.12g} доли")


# ---------------------------------------------------------------- commands
def cmd_check(a):
    angles = tuple(_rounded(value, a.digits) for value in a.angles)
    h, v = levels(*angles)
    print(
        f"theta1={fmt(angles[0], a.digits)} theta2={fmt(angles[1], a.digits)} "
        f"theta3={fmt(angles[2], a.digits)} theta4={fmt(angles[3], a.digits)} "
        f"-> PDH={fmt(float(h), a.digits)} PDV={fmt(float(v), a.digits)}"
    )


def cmd_alice(a):
    bob = tuple(a.bob)
    sols = solve(bob, a.target, a.theta2, role="alice")
    if not sols:
        sys.exit(
            "Решений нет: при заданных углах Боба и theta2 нулевой уровень не достижим."
        )
    shown_bob = tuple(_rounded(value, a.digits) for value in bob)
    print(
        f"Боб theta3={fmt(shown_bob[0], a.digits)} theta4={fmt(shown_bob[1], a.digits)}"
    )
    for k, (t1, t2) in enumerate(sols, start=1):
        shown_state = (_rounded(t1, a.digits), _rounded(t2, a.digits))
        h, v = levels(*shown_state, *shown_bob)
        print(
            f"{k}: theta1={fmt(shown_state[0], a.digits)} "
            f"theta2={fmt(shown_state[1], a.digits)} "
            f"PDH={fmt(float(h), a.digits)} PDV={fmt(float(v), a.digits)}"
        )


def cmd_bob(a):
    alice = tuple(a.alice)
    sols = solve(alice, a.target, a.theta4, role="bob")
    if not sols:
        sys.exit(
            "Решений нет: при заданных углах Алисы и theta4 нулевой уровень не достижим."
        )
    shown_alice = tuple(_rounded(value, a.digits) for value in alice)
    print(
        f"Алиса theta1={fmt(shown_alice[0], a.digits)} "
        f"theta2={fmt(shown_alice[1], a.digits)}"
    )
    for k, (t3, t4) in enumerate(sols, start=1):
        shown_bob = (_rounded(t3, a.digits), _rounded(t4, a.digits))
        h, v = levels(*shown_alice, *shown_bob)
        print(
            f"{k}: theta3={fmt(shown_bob[0], a.digits)} "
            f"theta4={fmt(shown_bob[1], a.digits)} "
            f"PDH={fmt(float(h), a.digits)} PDV={fmt(float(v), a.digits)}"
        )


def cmd_bb84(a):
    b0 = tuple(a.b0)
    a00 = _first_solution(
        solve(b0, "h0"), "не удалось подобрать состояние для базиса 0, бита 0"
    )
    a01 = _first_solution(
        solve(b0, "v0", second=a00[1]),
        "не удалось подобрать состояние для базиса 0, бита 1 при общем theta2",
    )

    if a.b1 is not None:
        if a.th3 is not None or a.root is not None:
            sys.exit("Используйте --b1 отдельно от --th3 и --root.")
        b1 = tuple(a.b1)
        fractions = [float(levels(*state, *b1)[0]) / FULL for state in (a00, a01)]
        if any(abs(value - 0.5) > 1e-8 for value in fractions):
            sys.exit(
                "--b1 не является равновероятным базисом: доли PDH для базиса 0 "
                f"равны {fractions[0]:.12g} и {fractions[1]:.12g}; "
                "ожидается 0.5 с допуском 1e-8."
            )
    else:
        th3 = 100.0 if a.th3 is None else a.th3
        roots = theta4_equal_split(a00, th3)
        if not roots:
            sys.exit(
                "Для такого theta3 не найдено равное деление. Попробуйте другое значение."
            )
        root_index = -1 if a.root is None else a.root
        if not -len(roots) <= root_index < len(roots):
            sys.exit(
                f"Найдено {len(roots)} значений theta4; допустимые --root: "
                f"{-len(roots)}..{len(roots) - 1}."
            )
        b1 = (th3, roots[root_index])

    a10 = _first_solution(
        solve(b1, "v0"), "не удалось подобрать состояние для базиса 1, бита 0"
    )
    a11 = _first_solution(
        solve(b1, "h0", second=a10[1]),
        "не удалось подобрать состояние для базиса 1, бита 1 при общем theta2",
    )
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    alice = {(0, 0): a00, (0, 1): a01, (1, 0): a10, (1, 1): a11}
    bob = {(0, 0): b0, (0, 1): b0, (1, 0): b1, (1, 1): b1}
    decoder = {(0, 0): (1, 0), (0, 1): (1, 0), (1, 0): (0, 1), (1, 1): (0, 1)}
    result = _protocol_payload("bb84", keys, alice, bob, decoder, a.digits)
    _emit_protocol(result, a.json_output, a.digits)


def cmd_goks(a):
    b00 = tuple(a.b00)
    sols = solve(b00, "h0")
    if not sols:
        sys.exit("Для указанных углов --bob не найдено состояние с PDH=0.")
    if not 1 <= a.sol <= len(sols):
        sys.exit(
            f"Решений для исходного состояния: {len(sols)}; допустимые --sol: 1..{len(sols)}."
        )
    t1_0, t2 = sols[a.sol - 1]
    step = -22.5 if a.flip else 22.5
    alice = {
        (0, 1): (t1_0, t2),
        (0, 0): (shift90(t1_0, step), t2),
        (1, 1): (shift90(t1_0, 2 * step), t2),
        (1, 0): (shift90(t1_0, 3 * step), t2),
    }
    bob = {(0, 0): b00}
    for basis, bit in ((0, 1), (1, 0), (1, 1)):
        sols_bob = solve(alice[(basis, 1 - bit)], "h0", role="bob")
        if not sols_bob:
            sys.exit(f"Не удалось подобрать Боба для базиса {basis}, бита {bit}.")
        bob[(basis, bit)] = sols_bob[0]

    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    decoder = {key: (key[1], "X") for key in keys}
    result = _protocol_payload("goks", keys, alice, bob, decoder, a.digits)
    _emit_protocol(result, a.json_output, a.digits)


def cmd_b92(a):
    b0 = tuple(a.b0)
    solutions = solve(b0, "h0")
    if not solutions:
        sys.exit("Для указанных углов --bob не найдено состояние с PDH=0.")
    t1_1, t2 = solutions[0]
    t1_0 = shift90(t1_1, 22.5 if a.flip else -22.5)
    alice = {(0, 0): (t1_0, t2), (0, 1): (t1_1, t2)}
    bob_bit1 = solve(alice[(0, 0)], "h0", role="bob")
    if not bob_bit1:
        sys.exit("Не удалось подобрать Боба для бита 1.")
    bob = {(0, 0): b0, (0, 1): bob_bit1[0]}
    keys = [(0, 0), (0, 1)]
    decoder = {key: (key[1], "X") for key in keys}
    result = _protocol_payload("b92", keys, alice, bob, decoder, a.digits)
    _emit_protocol(result, a.json_output, a.digits)


def _build_parser() -> tuple[
    argparse.ArgumentParser, dict[str, argparse.ArgumentParser]
]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    commands: dict[str, argparse.ArgumentParser] = {}
    s = sub.add_parser("check")
    commands["check"] = s
    s.add_argument(
        "angles", type=_finite_float, nargs=4, metavar=("T1", "T2", "T3", "T4")
    )
    s.add_argument("--digits", type=int, choices=range(1, 13), default=6)
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("alice")
    commands["alice"] = s
    s.add_argument(
        "--bob", type=_finite_float, nargs=2, required=True, metavar=("T3", "T4")
    )
    s.add_argument(
        "--target", choices=("h0", "v0"), default="h0", help="h0: PDH=0, v0: PDV=0"
    )
    s.add_argument(
        "--theta2",
        type=_finite_float,
        default=None,
        help="зафиксировать θ2 и искать только θ1",
    )
    s.add_argument("--digits", type=int, choices=range(1, 13), default=6)
    s.set_defaults(fn=cmd_alice)
    s = sub.add_parser("bob")
    commands["bob"] = s
    s.add_argument(
        "--alice", type=_finite_float, nargs=2, required=True, metavar=("T1", "T2")
    )
    s.add_argument(
        "--target", choices=("h0", "v0"), default="h0", help="h0: PDH=0, v0: PDV=0"
    )
    s.add_argument(
        "--theta4",
        type=_finite_float,
        default=None,
        help="зафиксировать θ4 и искать только θ3",
    )
    s.add_argument("--digits", type=int, choices=range(1, 13), default=6)
    s.set_defaults(fn=cmd_bob)
    s = sub.add_parser("bb84")
    commands["bb84"] = s
    s.add_argument(
        "--b0",
        "--bob",
        dest="b0",
        type=_finite_float,
        nargs=2,
        default=(65.0, 35.0),
        metavar=("T3", "T4"),
    )
    s.add_argument("--b1", type=_finite_float, nargs=2, metavar=("T3", "T4"))
    s.add_argument("--th3", type=_finite_float, default=None, help="θ3 второго базиса")
    s.add_argument(
        "--root", type=int, default=None, help="индекс theta4 второго базиса"
    )
    s.add_argument("--digits", type=int, choices=range(1, 13), default=6)
    s.add_argument("--json", action="store_true", dest="json_output")
    s.set_defaults(fn=cmd_bb84)
    s = sub.add_parser("goks")
    commands["goks"] = s
    s.add_argument(
        "--b00",
        "--bob",
        dest="b00",
        type=_finite_float,
        nargs=2,
        default=(48.0, 78.0),
        metavar=("T3", "T4"),
        help="произвольные углы Боба для (базис 0, бит 0)",
    )
    s.add_argument(
        "--sol", type=int, default=1, help="какое из решений для (базис 0, бит 1) брать"
    )
    s.add_argument(
        "--flip", action="store_true", help="обойти окружность в другую сторону"
    )
    s.add_argument("--digits", type=int, choices=range(1, 13), default=6)
    s.add_argument("--json", action="store_true", dest="json_output")
    s.set_defaults(fn=cmd_goks)
    s = sub.add_parser("b92")
    commands["b92"] = s
    s.add_argument(
        "--b0",
        "--bob",
        dest="b0",
        type=_finite_float,
        nargs=2,
        default=(32.0, 20.0),
        metavar=("T3", "T4"),
        help="произвольные углы Боба для ожидаемого бита 0",
    )
    s.add_argument(
        "--flip",
        action="store_true",
        help="состояние бита 0 по другую сторону от бита 1",
    )
    s.add_argument("--digits", type=int, choices=range(1, 13), default=6)
    s.add_argument("--json", action="store_true", dest="json_output")
    s.set_defaults(fn=cmd_b92)
    return p, commands


def protocol_main(protocol: str, argv: list[str] | None = None) -> None:
    if protocol not in {"bb84", "goks", "b92"}:
        raise ValueError("protocol must be bb84, goks, or b92")
    _, commands = _build_parser()
    parser = commands[protocol]
    parser.prog = Path(sys.argv[0]).name
    args = parser.parse_args(argv)
    args.fn(args)


def main(argv: list[str] | None = None) -> None:
    p, _ = _build_parser()
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
