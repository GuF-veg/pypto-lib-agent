# API Index

Every name exported by `pypto.language`, grouped by purpose. This page is
generated from the live module surface, so it doubles as the completeness
reference for the rest of the guide: a name that is not listed here is not
reachable as `pl.<name>`.

See [Syntax](02-syntax.md) for the language grammar and
[Tile Operations](06-tile-operations.md) /
[Tensor and System Operations](07-tensor-and-system-operations.md) for
semantics. Namespaces such as `pl.tensor` and `pl.system` are documented on
those pages.

## Types and annotations

| Name | Summary |
|---|---|
| `Array(*args: Any, **kwargs: Any) -> 'Array'` | On-core array wrapper. |
| `AsyncEvent(*, expr: pypto.pypto_core.ir.Expr \| None = None) -> None` | Handle to an in-flight asynchronous prefetch completion event. |
| `AsyncSession(*, expr: pypto.pypto_core.ir.Expr \| None = None) -> None` | Handle to the asynchronous session an [`AsyncEvent`][pypto.language.AsyncEvent] belongs to. |
| `ConstexprMarker()` | Annotation marking a parameter the compiler resolves at specialization time. |
| `DynVar(*args: Any, **kwargs: Any) -> 'Scalar'` | Dynamic shape variable for use in type annotations. |
| `InOut()` | Wrapper for InOut parameter direction in type annotations. |
| `MemRef(*args: Any, **kwargs: Any) -> None` | DSL-level memory reference accepting ``pl.Ptr`` bases and Scalar offsets. |
| `Out()` | Wrapper for Out parameter direction in type annotations. |
| `PrefetchAsyncContext(*, expr: pypto.pypto_core.ir.Expr \| None = None) -> None` | Handle to an asynchronous GM->L2 prefetch context. |
| `Ptr(*, expr: pypto.pypto_core.ir.Expr \| None = None) -> None` | DSL wrapper for an ``ir.PtrType``-valued expression. |
| `RUNTIME` | Retained marker asking for the behavior every scalar parameter now has. |
| `Scalar(*args: Any, **kwargs: Any) -> 'Scalar'` | Scalar type for PyPTO Language DSL. |
| `TaskId` | Scalar type for PyPTO Language DSL. |
| `Tensor(*args: Any, **kwargs: Any) -> ~TensorT` | Tensor type for PyPTO Language DSL. |
| `Tile(shape=None, dtype=None, expr: pypto.pypto_core.ir.Expr \| None = None, memref: 'MemRef \| None' = None, memory_space: 'MemorySpace \| None' = None, tile_view: 'TileView \| None' = None, _annotation_only: bool = False) -> 'Tile'` | Tile type for PyPTO Language DSL. |
| `Tuple(*args: Any, **kwargs: Any) -> 'Tuple'` | Tuple type for PyPTO Language DSL. |
| `constexpr(*args: Any, **kwargs: Any) -> Any` | Annotation marking a parameter the compiler resolves at specialization time. |
| `dynamic(name: str) -> pypto.language.typing.dynamic.DynVar` | Create a dynamic shape variable for type annotations. |

## Decorators, parsing and programs

| Name | Summary |
|---|---|
| `InlineFunction(name: str, func_def: ast.FunctionDef, param_names: list[str], source_file: str, source_lines: list[str], line_offset: int, col_offset: int, closure_vars: dict[str, typing.Any]) -> None` | Stores AST and metadata for a function to be inlined at call sites. |
| `JITFunction(func: 'Any', func_type: 'str \| None' = None, level: 'Any' = None, auto_scope: 'bool' = True, external_core_type: 'str \| None' = None, external_aic_source: 'str \| None' = None, external_aiv_source: 'str \| None' = None, external_dual_aiv_dispatch: 'bool' = False, external_include_dirs: 'tuple[str, ...]' = ()) -> 'None'` | A JIT-compiled function with shape specialization and caching. |
| `function(func: collections.abc.Callable[..., typing.Any] \| None = None, *, type: pypto.pypto_core.ir.FunctionType = FunctionType.Opaque, level: pypto.pypto_core.ir.Level \| None = None, role: pypto.pypto_core.ir.Role \| None = None, attrs: dict[str, typing.Any] \| None = None, auto_scope: bool = True, strict_ssa: bool = False, external_source: str \| pathlib.Path \| None = None) -> pypto.pypto_core.ir.Function \| collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], pypto.pypto_core.ir.Function]` | Decorator that parses a DSL function and returns IR Function. |
| `inline(func: collections.abc.Callable) -> pypto.language.parser.decorator.InlineFunction` | Decorator that captures a function for inlining at call sites. |
| `jit(func: 'Any' = None, *, auto_scope: 'bool' = True) -> 'Any'` | The ``pl.jit`` object. |
| `loads(filepath: str) -> pypto.pypto_core.ir.Function \| pypto.pypto_core.ir.Program` | Load a DSL function or program from a file. |
| `loads_program(filepath: str) -> pypto.pypto_core.ir.Program` | Load a DSL program from a file. |
| `parse(code: str, filename: str = '<string>', source_map: dict[int, tuple[str, int, int]] \| None = None) -> pypto.pypto_core.ir.Function \| pypto.pypto_core.ir.Program` | Parse a DSL function or program from a string. |
| `parse_program(code: str, filename: str = '<string>') -> pypto.pypto_core.ir.Program` | Parse a DSL program from a string. |
| `program(cls: type \| None = None, *, strict_ssa: bool = False) -> pypto.pypto_core.ir.Program \| collections.abc.Callable[[type], pypto.pypto_core.ir.Program]` | Decorator that parses a class with @pl.function methods into a Program. |

## Loops, scopes and control flow

| Name | Summary |
|---|---|
| `ScopeMode(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Dependency-tracking mode of a runtime scope (``SIMPLER_SCOPE``). |
| `at(level: 'ir.Level', role: 'ir.Role \| None' = None, *, optimizations: 'list[Optimization] \| None' = None, deps: 'list[Any] \| None' = None, no_dep_args: 'list[Any] \| None' = None, dumps: 'list[Any] \| None' = None, allow_early_resolve: 'bool' = False, predicate: 'Any' = None, name_hint: 'str' = '', windowize: 'bool' = False) -> 'AtContext'` | Mark a region of code for execution at a specific hierarchy level. |
| `cluster(*, name_hint: 'str' = '') -> 'ClusterContext'` | Mark a region of code as belonging to a Cluster execution context. |
| `cond(condition: 'CondArg') -> 'None'` | Specify the condition for a pl.while_() loop. |
| `const(value: 'int \| float', dtype: 'Any') -> 'Scalar'` | Create a typed constant with an explicit dtype. |
| `cross_core_slot(*, slot_num: 'int') -> 'CrossCoreSlot'` | Create a ``CrossCoreSlot`` optimization entry. |
| `func_attr(attrs: 'dict[str, Any]') -> 'None'` | Attach function-level attributes from inside the function body. |
| `graph(name: 'str') -> 'GraphContext'` | Mark a repeated region of orchestration as one recordable graph. |
| `manual_scope()` | Alias for ``pl.scope(mode=pl.ScopeMode.MANUAL)``. |
| `parallel(*args: 'RangeArg', init_values: 'tuple[Any, ...] \| None' = None) -> 'RangeIterator[Scalar] \| RangeIterator[tuple[Scalar, tuple[Any, ...]]]'` | Create a parallel range iterator for parallel for loops. |
| `pipeline(*args: 'RangeArg', stage: 'int', init_values: 'tuple[Any, ...] \| None' = None) -> 'RangeIterator[Scalar] \| RangeIterator[tuple[Scalar, tuple[Any, ...]]]'` | Create a software-pipelined loop iterator. |
| `range(*args: 'RangeArg', init_values: 'tuple[Any, ...] \| None' = None) -> 'RangeIterator[Scalar] \| RangeIterator[tuple[Scalar, tuple[Any, ...]]]'` | Create a range iterator for for loops. |
| `scope(mode: 'ScopeMode' = <ScopeMode.AUTO: 0>)` | Context manager marking a runtime scope (``SIMPLER_SCOPE``) region. |
| `split(mode: 'SplitMode', *, slot_num: 'int \| None' = None) -> 'Split'` | Create a ``Split`` optimization entry. |
| `split_aiv(n: 'int', *, mode: 'ir.SplitMode') -> 'SplitAivContext'` | Open an explicit AIV-split region as an SPMD-style loop. |
| `spmd(core_num: 'RangeArg', *, sync_start: 'bool' = False, name_hint: 'str' = '', optimizations: 'list[Optimization] \| None' = None, deps: 'list[Any] \| None' = None, allow_early_resolve: 'bool' = False, predicate: 'Any' = None) -> 'SpmdContext'` | Dispatch a kernel with SPMD (Single Program Multiple Data) multi-block execution. |
| `spmd_submit(*args: Any, **kwargs: Any) -> Any` | Launch a kernel as an SPMD task and capture its producer TaskId. |
| `static_assert(condition: 'Any', msg: 'str' = '') -> 'None'` | Assert a condition at compile time (parse time). |
| `static_print(*args: 'Any') -> 'None'` | Print compile-time information about IR objects. |
| `submit(*args: Any, **kwargs: Any) -> Any` | Submit a kernel and capture its producer TaskId. |
| `unroll(*args: 'RangeArg') -> 'RangeIterator[Scalar]'` | Create an unroll range iterator for compile-time loop unrolling. |
| `while_(*, init_values: 'tuple[ExprType, ...] \| None' = None) -> 'WhileIterator[tuple[ExprType, ...]]'` | Create a while iterator for while loops. |
| `yield_(*values: 'Any') -> 'Any \| tuple[Any, ...]'` | Yield values from a scope (for, if). |

## Tile-level operations

| Name | Summary |
|---|---|
| `MemRefType()` | Opaque sentinel type for ``MemRef``-typed variables in printed IR. |
| `addc(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Element-wise carry addition of three tiles. |
| `addsc(lhs: pypto.language.typing.tile.Tile, rhs: int \| float \| pypto.pypto_core.ir.Expr \| pypto.language.typing.scalar.Scalar, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Element-wise scalar carry addition. |
| `aic_gather(x: ~_SplitOperandT, span: pypto.pypto_core.ir.Span \| None = None) -> ~_SplitOperandT` | Hand a vector-produced operand to the cube (AIV -> AIC crossing). |
| `aiv_shard(x: ~_SplitOperandT, span: pypto.pypto_core.ir.Span \| None = None) -> ~_SplitOperandT` | Bring a cube-produced operand onto the AIV lane (AIC -> AIV crossing). |
| `cmps(lhs: pypto.language.typing.tile.Tile, rhs: int \| float \| pypto.pypto_core.ir.Expr \| pypto.language.typing.scalar.Scalar, cmp_type: int = 0) -> pypto.language.typing.tile.Tile` | Element-wise comparison of tile and scalar. |
| `create_tile(shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, target_memory: pypto.pypto_core.ir.MemorySpace \| None = None, transpose: bool \| None = None, *, flat_layout: bool \| None = None, compact: bool \| None = None) -> pypto.language.typing.tile.Tile` | Create a tile from a shape. |
| `gatherb(src: pypto.language.typing.tile.Tile, offset: pypto.language.typing.tile.Tile, *, output_dtype: int \| pypto.pypto_core.DataType \| None = None) -> pypto.language.typing.tile.Tile` | Gather 32-byte blocks from ``src`` by UINT32 byte offsets. |
| `gemv(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified) -> pypto.language.typing.tile.Tile` | General Matrix-Vector multiplication: C[1,N] = A[1,K] @ B[K,N]. |
| `gemv_acc(acc: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified, *, init_cond: bool \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr \| None = None) -> pypto.language.typing.tile.Tile` | GEMV with accumulation: C[1,N] += A[1,K] @ B[K,N]. |
| `gemv_bias(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile, acc_phase: pypto.pypto_core.ir.AccPhase = AccPhase.Unspecified) -> pypto.language.typing.tile.Tile` | GEMV with bias add: C[1,N] = A[1,K] @ B[K,N] + bias[1,N]. |
| `load(tensor: pypto.language.typing.tensor.Tensor, offsets: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], shapes: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], valid_shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr] \| None = None, target_memory: pypto.pypto_core.ir.MemorySpace \| None = None, clamp: bool = False, cache: pypto.pypto_core.ir.CachePolicy \| None = None) -> pypto.language.typing.tile.Tile` | Copy data from tensor to unified buffer (tile). |
| `lrelu(tile: pypto.language.typing.tile.Tile, slope: int \| float \| pypto.pypto_core.ir.Expr \| pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile` | Element-wise leaky ReLU with scalar slope. |
| `matmul_bias(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Matrix multiplication with bias add: C = lhs @ rhs + bias. |
| `matmul_mx_acc(acc: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, lhs_scale: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs_scale: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | MX block-scale matmul with accumulation. |
| `matmul_mx_bias(lhs: pypto.language.typing.tile.Tile, lhs_scale: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs_scale: pypto.language.typing.tile.Tile, bias: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | MX block-scale matmul with bias. |
| `max(lhs: pypto.language.typing.scalar.Scalar \| int \| pypto.pypto_core.ir.Expr, rhs: pypto.language.typing.scalar.Scalar \| int \| pypto.pypto_core.ir.Expr) -> pypto.language.typing.scalar.Scalar` | Scalar max of two values. |
| `maximums(lhs: pypto.language.typing.tile.Tile, rhs: int \| float \| pypto.pypto_core.ir.Expr \| pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile` | Element-wise maximum of tile and scalar. |
| `mgather(mem: pypto.language.typing.tensor.Tensor, idx: pypto.language.typing.tile.Tile \| pypto.language.typing.tensor.Tensor, coalesce: str \| int = 'row', *, gather_oob: str \| int = 'undefined', target_memory: pypto.pypto_core.ir.MemorySpace = MemorySpace.Vec, scratch: pypto.language.typing.tensor.Tensor \| None = None, valid_shape: collections.abc.Sequence[int] \| None = None) -> pypto.language.typing.tile.Tile` | Gather-load rows or elements from a GM tensor into a fresh Vec or Mat tile. |
| `min(lhs: pypto.language.typing.scalar.Scalar \| int \| pypto.pypto_core.ir.Expr, rhs: pypto.language.typing.scalar.Scalar \| int \| pypto.pypto_core.ir.Expr) -> pypto.language.typing.scalar.Scalar` | Scalar min of two values. |
| `minimums(lhs: pypto.language.typing.tile.Tile, rhs: int \| float \| pypto.pypto_core.ir.Expr \| pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile` | Element-wise minimum of tile and scalar. |
| `move(tile: pypto.language.typing.tile.Tile, target_memory: pypto.pypto_core.ir.MemorySpace, blayout: pypto.pypto_core.ir.TileLayout \| None = None, slayout: pypto.pypto_core.ir.TileLayout \| None = None) -> pypto.language.typing.tile.Tile` | Move tile between memory levels. |
| `mscatter(src: pypto.language.typing.tile.Tile, idx: pypto.language.typing.tile.Tile, output_tensor: ~_TensorT) -> ~_TensorT` | Scatter-store tile elements into a tensor at per-element indices. |
| `prelu(tile: pypto.language.typing.tile.Tile, slope: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Element-wise parametric ReLU of a tile. |
| `relu(tile: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Element-wise ReLU activation (max(0, x)). |
| `rem(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, high_precision: bool = False) -> pypto.language.typing.tile.Tile` | Element-wise remainder (modulo) of two tiles. |
| `rems(lhs: pypto.language.typing.tile.Tile, rhs: int \| float \| pypto.pypto_core.ir.Expr \| pypto.language.typing.scalar.Scalar, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Element-wise remainder (modulo) of tile and scalar. |
| `sel(mask: pypto.language.typing.tile.Tile, lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Per-element selection between two tiles using a predicate mask tile. |
| `sels(mask: pypto.language.typing.tile.Tile, src: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, scalar: int \| float \| pypto.pypto_core.ir.Expr \| pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tile.Tile` | Per-element selection between a source tile and a scalar. |
| `store(tile: pypto.language.typing.tile.Tile, offsets: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], output_tensor: ~_TensorT, shapes: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr] \| None = None, *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_, st_phase: pypto.pypto_core.ir.STPhase = STPhase.Unspecified, pre_quant: float \| None = None, pre_relu: bool = False) -> ~_TensorT` | Copy data from tile back to tensor. |
| `subc(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Element-wise carry subtraction of three tiles. |
| `subsc(lhs: pypto.language.typing.tile.Tile, rhs: int \| float \| pypto.pypto_core.ir.Expr \| pypto.language.typing.scalar.Scalar, rhs2: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Element-wise scalar carry subtraction. |
| `tmov_x2zz(src: pypto.language.typing.tile.Tile, tmp: pypto.language.typing.tile.Tile, *, group_axis: int = 1, dst_rows: int \| None = None, dst_cols: int \| None = None) -> pypto.language.typing.tile.Tile` | Exponent X-to-ZZ layout conversion (A5-only). |
| `tri(diagonal: int \| pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int], valid_shape: collections.abc.Sequence[int] \| None = None, dtype: pypto.pypto_core.DataType = int32, upper: bool = False) -> pypto.language.typing.tile.Tile` | Generate a lower- or upper-triangular mask tile. |

## Unified operations (dispatch on Tensor or Tile)

| Name | Summary |
|---|---|
| `abs(input: ~T) -> ~T` | Element-wise absolute value, dispatched by input type. |
| `add(lhs, rhs)` | Element-wise addition, dispatched by input type. |
| `and_(lhs, rhs)` | Element-wise bitwise AND, dispatched by input type. |
| `ands(lhs, rhs)` | Element-wise bitwise AND with a scalar, dispatched by input type. |
| `assemble(target, source, offset, *, atomic: pypto.pypto_core.ir.AtomicType = AtomicType.None_, pre_quant: float \| None = None, pre_relu: bool = False)` | Write ``source`` into ``target`` at ``offset``, dispatched by target type. |
| `batch_matmul(lhs: pypto.language.typing.tile.Tile, rhs: pypto.language.typing.tile.Tile) -> pypto.language.typing.tile.Tile` | Tile-only batched matrix multiplication. |
| `cast(input: pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile \| pypto.language.typing.scalar.Scalar, target_type: int \| pypto.pypto_core.DataType, mode: str \| int = 'round', *, saturation_mode: str \| int \| None = None) -> pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile \| pypto.language.typing.scalar.Scalar` | Type casting, dispatched by input type. |
| `cmp(lhs, rhs, cmp_type: int = 0)` | Element-wise comparison, dispatched by input type. |
| `col_argmax(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None)` | Column-wise argmax (per-column max index, int32), dispatched by input type. |
| `col_argmin(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None)` | Column-wise argmin (per-column min index, int32), dispatched by input type. |
| `col_expand(lhs: ~T, rhs: ~T) -> ~T` | Column-wise expansion, dispatched by input type. |
| `col_expand_add(lhs: ~T, rhs: ~T) -> ~T` | Column-wise broadcast addition, dispatched by input type. |
| `col_expand_div(lhs: ~T, rhs: ~T) -> ~T` | Column-wise broadcast division, dispatched by input type. |
| `col_expand_expdif(lhs: ~T, rhs: ~T) -> ~T` | Column-wise exp-diff (exp(lhs - rhs) with per-column scalar), dispatched by input type. |
| `col_expand_max(lhs: ~T, rhs: ~T) -> ~T` | Column-wise broadcast maximum, dispatched by input type. |
| `col_expand_min(lhs: ~T, rhs: ~T) -> ~T` | Column-wise broadcast minimum, dispatched by input type. |
| `col_expand_mul(lhs: ~T, rhs: ~T) -> ~T` | Column-wise broadcast multiplication, dispatched by input type. |
| `col_expand_sub(lhs: ~T, rhs: ~T) -> ~T` | Column-wise broadcast subtraction, dispatched by input type. |
| `col_max(input: ~T) -> ~T` | Column-wise max reduction, dispatched by input type. |
| `col_min(input: ~T) -> ~T` | Column-wise min reduction, dispatched by input type. |
| `col_prod(input: ~T) -> ~T` | Column-wise product reduction, dispatched by input type. |
| `col_sum(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None, *, is_binary: bool = False)` | Column-wise sum reduction, dispatched by input type. |
| `concat(src0: ~T, src1: ~T) -> ~T` | Column-wise concatenation, dispatched by input type. |
| `cos(input: ~T) -> ~T` | Element-wise cosine (input in radians), dispatched by input type. FP32 only. |
| `div(lhs, rhs, high_precision: bool = False)` | Element-wise division, dispatched by input type. |
| `exp(input: ~T) -> ~T` | Element-wise exponential, dispatched by input type. |
| `expands(target: pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile, scalar: int \| float \| pypto.language.typing.scalar.Scalar) -> pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile` | Expand scalar to target shape, dispatched by target type. **Unusable: no backend codegen on a2a3** (`No codegen registered for operation: tile.expands`). |
| `fillpad(value: ~T, pad_value: pypto.pypto_core.ir.PadValue \| int \| float = PadValue.zero) -> ~T` | Fill invalid elements, dispatched by input type. |
| `fillpad_expand(value: ~T, shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], pad_value: pypto.pypto_core.ir.PadValue \| int \| float = PadValue.zero) -> ~T` | Copy a smaller source into a larger destination, padding the rest. |
| `fmod(lhs, rhs, high_precision: bool = False)` | Element-wise truncating remainder, dispatched by input type. |
| `fmods(lhs, rhs)` | Element-wise truncating remainder with a scalar, dispatched by input type. |
| `gather_row(dst: ~T, src: pypto.language.typing.tensor.Tensor, dst_offset: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], src_offset: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], shapes: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], transpose: bool = False, *, valid_shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr] \| None = None) -> ~T` | Gather one GM row into a sub-region of an on-chip accumulator (DPS). |
| `log(input: ~T, high_precision: bool = False) -> ~T` | Element-wise natural logarithm, dispatched by input type. |
| `matmul(lhs: ~T, rhs: ~T, out_dtype: int \| pypto.pypto_core.DataType \| None = None, a_trans: bool = False, b_trans: bool = False, c_matrix_nz: bool = False) -> ~T` | Matrix multiplication, dispatched by input type. |
| `matmul_acc(acc: ~T, lhs: ~T, rhs: ~T, a_trans: bool = False, b_trans: bool = False, init_cond: bool \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr \| None = None) -> ~T` | Matrix multiplication with accumulation, dispatched by input type. |
| `matmul_mx(lhs, lhs_scale, rhs, rhs_scale)` | MXFP8 matrix multiplication, dispatched for tensors and tiles. |
| `maximum(lhs, rhs)` | Element-wise maximum, dispatched by input type. |
| `minimum(lhs, rhs)` | Element-wise minimum, dispatched by input type. |
| `mrgsort(src0: ~T, src1: Optional[~T] = None, src2: Optional[~T] = None, src3: Optional[~T] = None, tmp: Optional[~T] = None, *, exhausted: bool = False, block_len: int \| pypto.language.typing.scalar.Scalar \| None = None) -> ~T` | Merge sort — format1 (single-list) or format2 (2-4 way), dispatched by input type. |
| `mul(lhs, rhs)` | Element-wise multiplication, dispatched by input type. |
| `neg(input: ~T) -> ~T` | Element-wise negation, dispatched by input type. |
| `not_(input: ~T) -> ~T` | Element-wise bitwise NOT, dispatched by input type (int16/uint16 only). |
| `or_(lhs, rhs)` | Element-wise bitwise OR, dispatched by input type. |
| `ors(lhs, rhs)` | Element-wise bitwise OR with a scalar, dispatched by input type. |
| `part_add(lhs, rhs)` | Partial element-wise add, dispatched by input type. |
| `part_max(lhs, rhs)` | Partial element-wise max, dispatched by input type. |
| `part_min(lhs, rhs)` | Partial element-wise min, dispatched by input type. |
| `part_mul(lhs, rhs)` | Partial element-wise multiply, dispatched by input type. |
| `quant_mx(src, *, group_axis: int, dtype: pypto.pypto_core.DataType = fp8e4m3fn)` | MXFP8 quantization, dispatched for GM tensors and tiles. |
| `read(src: pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile, offset: int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr \| collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr]) -> pypto.language.typing.scalar.Scalar` | Read a scalar value at given indices, dispatched by source type. |
| `recip(input: ~T, high_precision: bool = False) -> ~T` | Element-wise reciprocal (1/x), dispatched by input type. |
| `reinterpret_view(data: ~T, dtype: pypto.pypto_core.DataType, *, shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr] \| None = None) -> ~T` | Reinterpret the same bytes with a different dtype. |
| `reshape(input: ~T, shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr]) -> ~T` | Reshape operation, dispatched by input type. |
| `row_argmax(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None)` | Row-wise argmax (per-row max index, int32), dispatched by input type. |
| `row_argmin(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None)` | Row-wise argmin (per-row min index, int32), dispatched by input type. |
| `row_expand(lhs: ~T, rhs: ~T) -> ~T` | Row-wise expansion, dispatched by input type. |
| `row_expand_add(lhs, rhs, tmp: pypto.language.typing.tile.Tile \| None = None)` | Row-wise broadcast addition; ``tmp`` is available only for Tile inputs. |
| `row_expand_div(lhs: ~T, rhs: ~T) -> ~T` | Row-wise broadcast division, dispatched by input type. |
| `row_expand_expdif(lhs: ~T, rhs: ~T) -> ~T` | Row-wise exp-diff (exp(lhs - rhs) with per-row scalar), dispatched by input type. |
| `row_expand_max(lhs: ~T, rhs: ~T) -> ~T` | Row-wise broadcast maximum, dispatched by input type. |
| `row_expand_min(lhs: ~T, rhs: ~T) -> ~T` | Row-wise broadcast minimum, dispatched by input type. |
| `row_expand_mul(lhs: ~T, rhs: ~T) -> ~T` | Row-wise broadcast multiplication, dispatched by input type. |
| `row_expand_sub(lhs: ~T, rhs: ~T) -> ~T` | Row-wise broadcast subtraction, dispatched by input type. |
| `row_max(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None)` | Row-wise max reduction, dispatched by input type. |
| `row_min(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None)` | Row-wise min reduction, dispatched by input type. |
| `row_prod(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None)` | Row-wise product reduction, dispatched by input type. |
| `row_sum(input, tmp_tile: pypto.language.typing.tile.Tile \| None = None)` | Row-wise sum reduction, dispatched by input type. |
| `rsqrt(input, high_precision: bool = False)` | Element-wise reciprocal square root, dispatched by input type. |
| `scatter_update(input: ~T, *args: Any, **kwargs: Any) -> ~T` | Update rows at positions given by a 2D index, dispatched by input type. |
| `set_validshape(input, valid_rows, valid_cols)` | Update valid-shape metadata without data movement, dispatched by input type. |
| `shl(lhs, rhs)` | Element-wise bitwise left shift, dispatched by input type. |
| `shls(lhs, rhs)` | Element-wise bitwise left shift by a scalar, dispatched by input type. |
| `shr(lhs, rhs)` | Element-wise bitwise right shift, dispatched by input type. |
| `shrs(lhs, rhs)` | Element-wise bitwise right shift by a scalar, dispatched by input type. |
| `sin(input: ~T) -> ~T` | Element-wise sine (input in radians), dispatched by input type. FP32 only. |
| `slice(input: ~T, shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], offset: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], valid_shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr] \| None = None, drop_dims: collections.abc.Sequence[int \| pypto.pypto_core.ir.Expr] \| None = None, pad_value: pypto.pypto_core.ir.PadValue \| int \| float \| None = None, clamp: bool = False) -> ~T` | Slice operation, dispatched by input type. |
| `sort32(src: ~T, idx: ~T) -> ~T` | Sort fixed 32-element blocks, permuting ``idx`` alongside ``src``. |
| `sqrt(input: ~T) -> ~T` | Element-wise square root, dispatched by input type. |
| `sub(lhs, rhs)` | Element-wise subtraction, dispatched by input type. |
| `transpose(input: ~T, axis1: int, axis2: int) -> ~T` | Transpose operation, dispatched by input type. |
| `write(dst: pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile, offset: int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr \| collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], value: pypto.language.typing.scalar.Scalar) -> pypto.pypto_core.ir.Expr` | Write a scalar value to a tensor or tile at given indices. |
| `xor(lhs, rhs, tmp=None)` | Element-wise bitwise XOR, dispatched by input type. |
| `xors(lhs, rhs, tmp=None)` | Element-wise bitwise XOR with a scalar, dispatched by input type. |

## Tensor-level operations

| Name | Summary |
|---|---|
| `arange(start: int \| pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType = int32, descending: bool = False) -> pypto.language.typing.tensor.Tensor` | Generate a contiguous integer sequence into a tensor. |
| `create_l1(shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, transpose: bool = False) -> pypto.language.typing.tensor.Tensor` | Create an on-chip (L1/Mat) accumulator for a kernel-driven paged gather. |
| `create_tensor(shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, layout: pypto.pypto_core.ir.TensorLayout = TensorLayout.ND, manual_dep: bool = False, init_value: int \| float \| None = None) -> pypto.language.typing.tensor.Tensor` | Create a new tensor with specified shape and dtype. |
| `dim(tensor: pypto.language.typing.tensor.Tensor, axis: int \| pypto.pypto_core.ir.ConstInt) -> pypto.language.typing.scalar.Scalar` | Extract a shape dimension from a tensor as a scalar value. |
| `dump_tag(tensor: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor` | Mark a tensor for selective dump within the enclosing orchestration. |
| `expand_clone(src: pypto.language.typing.tensor.Tensor, target: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor` | Clone and expand input to target shape. |
| `full(shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType, value: int \| float) -> pypto.language.typing.tensor.Tensor` | Create a tensor of specified shape filled with a constant value. |
| `gather(input: pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile, dim: int \| pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile \| None = None, index: pypto.language.typing.tensor.Tensor \| pypto.language.typing.tile.Tile \| None = None, *, mask_pattern: int \| None = None, output_dtype: int \| pypto.pypto_core.DataType \| None = None, kvalue: int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr \| None = None, cmp_mode: str \| int \| None = None, out_cols: int \| None = None, offset: int = 0, count_dtype: int \| pypto.pypto_core.DataType \| None = None) -> pypto.language.typing.tensor.Tensor \| tuple[pypto.language.typing.tensor.Tensor, pypto.language.typing.tensor.Tensor]` | Gather elements of ``input`` — flat / axis / mask / compare form. |
| `get_block_idx() -> pypto.language.typing.scalar.Scalar` | Get the current block index (tensor-scope alias of ``pl.tile.get_block_idx``). |
| `get_block_num() -> pypto.language.typing.scalar.Scalar` | Get the total number of blocks in the current SPMD task. |
| `get_subblock_idx() -> pypto.language.typing.scalar.Scalar` | Get the current sub-block (vector core) index (tensor-scope alias of ``pl.tile.get_subblock_idx``). |
| `no_dep(tensor: pypto.language.typing.tensor.Tensor) -> pypto.language.typing.tensor.Tensor` | Mark a kernel-call argument as no-dependency (caller-site override). |
| `paged_gather(src: pypto.language.typing.tensor.Tensor, indices: pypto.language.typing.tensor.Tensor, block_table: pypto.language.typing.tensor.Tensor, block_size: int, size: int, max_indices: int, *, space: pypto.pypto_core.ir.MemorySpace = MemorySpace.Mat, col_off: int = 0, is_trans: bool = False, is_b_matrix: bool = False) -> pypto.language.typing.tensor.Tensor` | Paged gather directly into an on-chip buffer (L1 by default, or UB). |
| `random(key0: int \| pypto.language.typing.scalar.Scalar, key1: int \| pypto.language.typing.scalar.Scalar, counter0: int \| pypto.language.typing.scalar.Scalar, counter1: int \| pypto.language.typing.scalar.Scalar, counter2: int \| pypto.language.typing.scalar.Scalar, counter3: int \| pypto.language.typing.scalar.Scalar, shape: collections.abc.Sequence[int \| pypto.language.typing.scalar.Scalar \| pypto.pypto_core.ir.Expr], dtype: pypto.pypto_core.DataType = uint32, rounds: int = 10) -> pypto.language.typing.tensor.Tensor` | Generate counter-based pseudo-random values into a tensor. |
| `scatter(input: pypto.language.typing.tensor.Tensor, dim: int \| None = None, index: pypto.language.typing.tensor.Tensor \| None = None, src: pypto.language.typing.tensor.Tensor \| None = None, *, mask_pattern: int \| None = None, dst: pypto.language.typing.tensor.Tensor \| None = None) -> pypto.language.typing.tensor.Tensor` | Scatter elements of ``src`` into ``input`` (tensor-level) — index or mask form. |
| `set_cache_policy(tensor: pypto.language.typing.tensor.Tensor, policy: pypto.pypto_core.ir.CachePolicy) -> None` | Declare the GM cache-access policy for every read of ``tensor`` in this scope. |

## System and cross-core operations

| Name | Summary |
|---|---|
| `import_peer_buffer(*, name: str, peer_func: str, span: pypto.pypto_core.ir.Span \| None = None) -> pypto.language.typing.scalar.Scalar` | Import a buffer from a peer function in the same group. |
| `reserve_buffer(*, name: str, size: int, base: int = -1, span: pypto.pypto_core.ir.Span \| None = None) -> pypto.language.typing.scalar.Scalar` | Reserve a named buffer for cross-core communication. |
| `tfree_to_aic(tile: pypto.language.typing.tile.Tile, span: pypto.pypto_core.ir.Span \| None = None, *, split: int \| None = None, id: int \| None = None) -> pypto.pypto_core.ir.Call` | Release ring buffer slot back to AIC producer. |
| `tfree_to_aiv(tile: pypto.language.typing.tile.Tile, span: pypto.pypto_core.ir.Span \| None = None, *, split: int \| None = None, id: int \| None = None) -> pypto.pypto_core.ir.Call` | Release ring buffer slot back to AIV producer. |
| `tpop_from_aic(*, shape: list[int] \| None = None, dtype: pypto.pypto_core.DataType \| None = None, split: int = 0, lane_stride: int \| None = None, id: int \| None = None, span: pypto.pypto_core.ir.Span \| None = None) -> pypto.language.typing.tile.Tile` | Pop tile data from AIC cross-core pipe into AIV. |
| `tpop_from_aiv(*, shape: list[int] \| None = None, dtype: pypto.pypto_core.DataType \| None = None, split: int = 0, id: int \| None = None, span: pypto.pypto_core.ir.Span \| None = None) -> pypto.language.typing.tile.Tile` | Pop tile data from AIV cross-core pipe into AIC. |
| `tpush_to_aic(tile: pypto.language.typing.tile.Tile, *, split: int, id: int \| None = None, span: pypto.pypto_core.ir.Span \| None = None) -> pypto.pypto_core.ir.Call` | Push tile data from AIV to AIC via cross-core pipe. |
| `tpush_to_aiv(tile: pypto.language.typing.tile.Tile, *, split: int, lane_stride: int \| None = None, id: int \| None = None, span: pypto.pypto_core.ir.Span \| None = None) -> pypto.pypto_core.ir.Call` | Push tile data from AIC to AIV via cross-core pipe. |

## Enums, descriptors and data types

| Name | Summary |
|---|---|
| `AccPhase(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Producer-side unit-flag phase for GEMV accumulator operations |
| `AtomicType(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Combine mode for global-memory writes — pld.tensor.put (TPUT) and tile.store (TSTORE) |
| `BF16` | Data type representation for PyPTO tensors and operations |
| `BOOL` | Data type representation for PyPTO tensors and operations |
| `CachePolicy(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | GM cache-access policy declared for a tensor read |
| `CompactMode(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Partial-tile compact mode enumeration |
| `DN` | DN layout |
| `FP16` | Data type representation for PyPTO tensors and operations |
| `FP32` | Data type representation for PyPTO tensors and operations |
| `FP4` | Data type representation for PyPTO tensors and operations |
| `FP4E2M1X2` | Data type representation for PyPTO tensors and operations |
| `FP8E4M3FN` | Data type representation for PyPTO tensors and operations |
| `FP8E5M2` | Data type representation for PyPTO tensors and operations |
| `FP8E8M0` | Data type representation for PyPTO tensors and operations |
| `ForKind(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | For loop kind classification |
| `FunctionType(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Function type classification |
| `HF4` | Data type representation for PyPTO tensors and operations |
| `HF8` | Data type representation for PyPTO tensors and operations |
| `INDEX` | Data type representation for PyPTO tensors and operations |
| `INT16` | Data type representation for PyPTO tensors and operations |
| `INT32` | Data type representation for PyPTO tensors and operations |
| `INT4` | Data type representation for PyPTO tensors and operations |
| `INT64` | Data type representation for PyPTO tensors and operations |
| `INT8` | Data type representation for PyPTO tensors and operations |
| `Level(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Hierarchy level in the Linqu machine model |
| `MX_A_ZZ` | MX Left/A scale GM pack (ZZ) |
| `MX_B_NN` | MX Right/B scale GM pack (NN) |
| `Mem(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Memory space enumeration |
| `MemorySpace(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Memory space enumeration |
| `ND` | ND layout |
| `NZ` | NZ layout |
| `PadValue(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Pad mode enumeration for tile/tensor views |
| `PipeType(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Pipeline type enumeration |
| `Role(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Function role at L3-L7 hierarchy levels |
| `STPhase(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Consumer-side unit-flag phase for tile.store |
| `SplitMode(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Split mode for cross-core data transfer |
| `TASK_ID` | Data type representation for PyPTO tensors and operations |
| `TensorLayout(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Tensor layout enumeration |
| `TileLayout(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Tile layout enumeration |
| `UINT16` | Data type representation for PyPTO tensors and operations |
| `UINT32` | Data type representation for PyPTO tensors and operations |
| `UINT4` | Data type representation for PyPTO tensors and operations |
| `UINT64` | Data type representation for PyPTO tensors and operations |
| `UINT8` | Data type representation for PyPTO tensors and operations |

## Namespaces and helpers

| Name | Summary |
|---|---|
| `AUTO` | int([x]) -> integer |
| `IntLike` | Represent a PEP 604 union type |
| `KernelType(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Which generated kernel an op belongs to. |
| `SyncAllMode(value, names=None, *, module=None, qualname=None, type=None, start=1, boundary=None)` | Barrier implementation selected by ``system.syncall``. |
| `TensorView(stride: collections.abc.Sequence[pypto.pypto_core.ir.Expr \| int] \| None = None, layout: pypto.pypto_core.ir.TensorLayout \| None = None, valid_shape: collections.abc.Sequence[pypto.pypto_core.ir.Expr \| int] \| None = None, pad: pypto.pypto_core.ir.PadValue = PadValue.null) -> '_TensorViewBase'` | TensorView factory: accepts Expr or int in stride/valid_shape. |
| `TileView(valid_shape: collections.abc.Sequence[pypto.pypto_core.ir.Expr \| int] \| None = None, stride: collections.abc.Sequence[pypto.pypto_core.ir.Expr \| int] \| None = None, start_offset: pypto.pypto_core.ir.Expr \| int \| None = None, blayout: pypto.pypto_core.ir.TileLayout = TileLayout.row_major, slayout: pypto.pypto_core.ir.TileLayout = TileLayout.none_box, fractal: int = 512, pad: pypto.pypto_core.ir.PadValue = PadValue.null, compact: pypto.pypto_core.ir.CompactMode = CompactMode.null) -> '_TileViewBase'` | TileView factory: accepts Expr or int in valid_shape/stride. |
| `adir` | Per-call-site direction markers for PyPTO Language DSL. |
| `aic_initialize_pipe(c2v_consumer_buf: pypto.pypto_core.ir.Expr \| int \| float \| pypto.ir.op.system_ops._UnwrapsToExpr = 0, v2c_consumer_buf: pypto.pypto_core.ir.Expr \| int \| float \| pypto.ir.op.system_ops._UnwrapsToExpr = 0, *, dir_mask: int, slot_size: int, slot_num: int \| None = None, local_slot_num: int \| None = None, id: int \| None = None, span: pypto.pypto_core.ir.Span \| None = None) -> pypto.pypto_core.ir.Call` | Initialize cross-core pipe on AIC side. |
| `aiv_initialize_pipe(c2v_consumer_buf: pypto.pypto_core.ir.Expr \| int \| float \| pypto.ir.op.system_ops._UnwrapsToExpr = 0, v2c_consumer_buf: pypto.pypto_core.ir.Expr \| int \| float \| pypto.ir.op.system_ops._UnwrapsToExpr = 0, *, dir_mask: int, slot_size: int, slot_num: int \| None = None, local_slot_num: int \| None = None, id: int \| None = None, span: pypto.pypto_core.ir.Span \| None = None) -> pypto.pypto_core.ir.Call` | Initialize cross-core pipe on AIV side. |
| `array` | Array operations for PyPTO Language DSL. |
| `optimizations` | Optimization config entries for ``pl.at(..., optimizations=[...])``. |
| `parser` | Language Parser module for converting high-level DSL code to IR structures. |
| `prefetch` | ``pl.prefetch.*`` — asynchronous GM->L2 prefetch operations. |
| `system` | System operations for PyPTO Language DSL. |
| `tensor` | Tensor operations for PyPTO Language DSL. |
| `tile` | Tile operations for PyPTO Language DSL. |

## Coverage

252 exported names are listed above. The page is produced by
introspecting `pypto.language` at the revision recorded in the
[overview](index.md#verification-baseline); regenerate it rather than
editing it by hand.
