"""Load the unchanged official floating point small architecture definitions.

Apache-2.0 source remains under third_party with its copyright and LICENSE.
Only unused imports and the BaseModel convenience wrapper are omitted. The
wrapper adds __str__ parameter reporting but no forward or parameter behavior.
No quantization, architecture search, or checkpoint is implied.
"""
import ast
from pathlib import Path
import torch.nn as nn
from math import ceil

SOURCE = Path(__file__).resolve().parents[1] / 'third_party/q-ppg-main/precision_search/model/TimePPG_float.py'

def build_model():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'), filename=str(SOURCE))
    definitions = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
    scope = {'nn': nn, 'ceil': ceil, 'BaseModel': nn.Module}
    exec(compile(definitions, str(SOURCE), 'exec'), scope)
    return scope['TimePPGfloat_small']()
