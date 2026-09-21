# Task 1: Intents Implementation Report

## Implementation Summary

Created the `Intents` frozen dataclass in `/Users/keshavgarg/Desktop/music-tech/mixing/src/mixengine/core/intents.py` and comprehensive test suite in `/Users/keshavgarg/Desktop/music-tech/mixing/tests/test_intents.py`.

### Critical Bug Fix Applied

The brief's code declared `AUTO: "Intents"` directly inside the @dataclass, which would incorrectly make it a required dataclass field. Applied the controller's ruling by declaring it as a ClassVar instead:

```python
from typing import Any, ClassVar, Dict, Optional
...
    AUTO: ClassVar["Intents"]
```

This ensures:
- `Intents()` creates a valid instance with all None values
- `asdict(Intents.AUTO)` only includes the 9 actual fields (not AUTO)
- `test_auto_has_every_field_none` passes as expected

### Implementation Details

**File: `src/mixengine/core/intents.py`**
- 115 lines total
- Frozen dataclass with 9 Optional fields: vocal_state, relationship, key, bpm, tune, timing, space, separate, loudness
- Three validation helper functions:
  - `_choice()`: validates against allowed tuples (vocal_state, relationship, space, separate)
  - `_strength()`: validates floats in [0, 1] or special values (auto, off)
  - `_number()`: validates floats within range (e.g., bpm 40-220, loudness -30 to -3)
- Static method `from_dict(Optional[Dict])` with full input validation
- Instance method `to_dict()` returns dataclass fields as dict
- Property `is_all_auto` checks if all fields are None
- Class variable `Intents.AUTO` initialized as singleton with all None values

**File: `tests/test_intents.py`**
- 60 lines, 8 test methods covering:
  - AUTO sentinel correctness
  - "auto" string to None conversion
  - Field value parsing (including type coercion)
  - Strength field special handling (off → 0.0, numeric values, range validation)
  - None dict equivalence
  - Invalid input rejection (unknown enum values, out-of-range values)
  - Round-trip serialization correctness

## Test Results

```
Command: .venv/bin/python -m pytest tests/test_intents.py -v
Result: 8 passed in 3.38s

Tests:
  ✓ test_auto_has_every_field_none
  ✓ test_from_dict_treats_auto_string_as_none
  ✓ test_from_dict_reads_values
  ✓ test_off_and_numeric_strength_both_become_floats
  ✓ test_none_dict_is_all_auto
  ✓ test_rejects_unknown_vocal_state
  ✓ test_rejects_strength_out_of_range
  ✓ test_round_trip
```

## Linting & Type Checking

```
Command: .venv/bin/ruff check src/mixengine/core/intents.py tests/test_intents.py
Result: All checks passed!

Command: .venv/bin/mypy src/mixengine/core/intents.py
Result: Success: no issues found in 1 source file
```

## Git Commit

Commit SHA: `d273151` (rebuild/intake-policy-frontend branch)

Message:
```
Add Intents: the facts measurement cannot establish

Provenance is not measurable. Whether a vocal was recorded to this
beat, and whether its pitch sits on the grid by intent, are things
only the person who made the recording knows. Intents carry them.
```

## Self-Review Findings

Reviewed the committed diff (`git diff HEAD~1`). All observations:

1. **ClassVar Fix**: Correctly applied. AUTO is now a class variable, not a dataclass field. Verified by test_auto_has_every_field_none which iterates only over the 9 data fields.

2. **Field Names & Types**: All 9 fields match the brief exactly, all Optional[str] or Optional[float] as specified.

3. **Validation Constants**: All four validation tuples match verbatim:
   - VOCAL_STATES = ("raw", "tuned", "finished")
   - RELATIONSHIPS = ("locked", "free")
   - SPACES = ("keep", "match", "add")
   - SEPARATIONS = ("never", "always")

4. **Validation Messages**: All error messages match the brief's exact format:
   - Choice fields: "must be one of ..." with 'auto' prepended
   - Strength fields: "must be auto, off, or 0-1" / "must be within 0-1"
   - Number fields: "must be a number" / "must be within X-Y"

5. **Helper Functions Logic**: 
   - _choice: Correctly treats None, "", and "auto" as None; validates against allowed tuple
   - _strength: Correctly converts "off" to 0.0, parses string/number, validates [0, 1]
   - _number: Correctly handles None/"auto"/"", validates range

6. **from_dict Method**: 
   - Returns empty Intents() for None or falsy dict (test_none_dict_is_all_auto)
   - Properly destructures and validates each field
   - Key handling uses strip().lower() on non-auto values

7. **to_dict Method**: Uses asdict(self), returns only the 9 fields (not AUTO) ✓

8. **is_all_auto Property**: Correctly iterates fields() and checks all are None

9. **Module-Level Initialization**: Intents.AUTO = Intents() placed after class definition ✓

10. **Python 3.9 Compatibility**: 
    - ClassVar imported from typing ✓
    - Dict and Optional from typing ✓
    - No match statements, no | unions, no builtin generics at runtime ✓
    - Field type annotations use Optional[...] not ... | None ✓

11. **Test Suite**: All 8 tests from the brief included verbatim, correct import preamble

12. **Code Style**: Consistent with project conventions, follows brief's exact code style

## Concerns

None. Implementation is complete, fully tested, passes all linting, and the critical ClassVar bug fix was correctly applied.

## Fix Round 1: Review Findings

### Finding 1: Unhandled TypeError in `_strength`

**Issue**: Line 51 (`number = float(value)`) was not protected by try/except, so invalid types (e.g., lists) raised `TypeError` instead of the promised `ValueError`.

**Fix Applied**: 
- Wrapped `number = float(value)` in `try/except (TypeError, ValueError)` block
- Re-raises as `ValueError` with consistent error message matching existing pattern in `_number`
- Bool values now explicitly rejected before float conversion to prevent silent acceptance

**Test Coverage**: `test_rejects_list_as_strength_with_valueerror` confirms that `Intents.from_dict({"tune": [1, 2]})` raises `ValueError` (not `TypeError`)

### Finding 2: Bool Silently Accepted in Strength and Number Fields

**Issue**: `isinstance(True, int)` is `True` in Python, so bool values bypassed the string branch and reached `float()` conversion. This allowed `Intents.from_dict({"tune": True})` to silently return `tune=1.0`.

**Fix Applied**:
- Added explicit `isinstance(value, bool)` check at the start of both `_strength` and `_number`
- Bool values now raise `ValueError` with descriptive message before reaching float conversion
- Error messages match existing patterns ("must be auto, off, or 0-1" for strength; "must be a number" for number)

**Test Coverage**: 
- `test_rejects_bool_as_strength` confirms `Intents.from_dict({"tune": True})` raises `ValueError`
- `test_rejects_bool_as_number` confirms `Intents.from_dict({"bpm": True})` raises `ValueError`

### Test Results

```
Command: .venv/bin/python -m pytest tests/test_intents.py -v
Result: 11 passed in 3.66s

Tests (8 original + 3 new):
  ✓ test_auto_has_every_field_none
  ✓ test_from_dict_treats_auto_string_as_none
  ✓ test_from_dict_reads_values
  ✓ test_off_and_numeric_strength_both_become_floats
  ✓ test_none_dict_is_all_auto
  ✓ test_rejects_unknown_vocal_state
  ✓ test_rejects_strength_out_of_range
  ✓ test_round_trip
  ✓ test_rejects_list_as_strength_with_valueerror (NEW)
  ✓ test_rejects_bool_as_strength (NEW)
  ✓ test_rejects_bool_as_number (NEW)
```

### Linting & Type Checking

```
Command: .venv/bin/ruff check src/mixengine/core/intents.py tests/test_intents.py
Result: All checks passed!

Command: .venv/bin/mypy src/mixengine/core/intents.py tests/test_intents.py
Result: Success: no issues found in 2 source files
```

### Git Commit

Commit SHA: `096ada4` (rebuild/intake-policy-frontend branch)

Message:
```
Fix two validation gaps in _strength and _number

Finding 1: _strength did not protect float() conversion from TypeError.
Wrapped float(value) in try/except (TypeError, ValueError) and re-raise
as ValueError, matching the error-handling pattern in _number.

Finding 2: bool values silently accepted in both _strength and _number.
isinstance(True, int) is True in Python, so bools bypass the string
branch and reach float() conversion. Added explicit bool rejection in
both functions before float conversion.
```
