# Gramática Libre de Contexto — Analizador Sintáctico (Triton GPU Kernel)

**Equipo 13 · TC3002B Gpo 501 · Fase II: Análisis Sintáctico**

Alcance: subconjunto *moderado* y realista de kernels Triton. CFG inambigua por
construcción (precedencia y asociatividad codificadas estructuralmente en la
cascada de expresiones). Diseñada para traducirse directamente a `yacc`.

---

## 1. Terminales (tokens)

### 1.1 Tokens existentes (del analizador léxico, IDs 1–62)

| Categoría | Terminales |
|---|---|
| Palabras reservadas | `DEF FOR IF ELSE ELIF WHILE IMPORT AS IN TRUE FALSE NONE RETURN BREAK CONTINUE PASS AND OR NOT IS ASSERT FROM TRY EXCEPT GLOBAL RAISE DEL` |
| Identificadores y literales | `IDENTIFIER SCIENTIFIC FLOAT INT STRING` |
| Operadores aritméticos | `PLUS MINUS MULTIPLY DIVIDE POWER FLOOR_DIV MODULO` |
| Comparación / asignación | `EQUAL LESS_THAN GREATER_THAN NOT_EQUAL LESS_EQUAL GREATER_EQUAL DOUBLE_EQUAL` |
| Bitwise | `BIT_AND BIT_OR BIT_NOT BIT_XOR LEFT_SHIFT RIGHT_SHIFT` |
| Delimitadores | `DOT COMMA COLON AT_SIGN OPEN_PAREN CLOSE_PAREN OPEN_BRACKET CLOSE_BRACKET OPEN_BRACE CLOSE_BRACE` |

### 1.2 Tokens NUEVOS requeridos por el parser (modificación al scanner)

| Token | ID propuesto | Significado |
|---|---|---|
| `NEWLINE` | 63 | Fin de línea lógica (separa sentencias) |
| `INDENT`  | 64 | Aumento de nivel de sangría (abre bloque) |
| `DEDENT`  | 65 | Disminución de nivel de sangría (cierra bloque) |

> Estos tres tokens son la razón por la que el scanner debe modificarse: deja de
> descartar `\n` y la sangría, y mantiene una pila de indentación para emitir
> `INDENT`/`DEDENT`. (Sección II.3.b de la rúbrica: "modify the scanner".)

**Convenciones de notación:**
- MAYÚSCULAS = terminal (token).
- minúsculas = no terminal.
- `[ X ]` = opcional (0 o 1).
- `ε` = producción vacía (epsilon).
- Símbolo inicial: **`program`**.

---

## 2. Estructura del programa y sentencias

```
program        : NEWLINE program          /* tolera líneas en blanco iniciales */
               | stmt_list

stmt_list      : stmt_list statement
               | statement

statement      : simple_stmt
               | compound_stmt
```

### 2.1 Sentencias simples (terminan en NEWLINE)

```
simple_stmt    : small_stmt NEWLINE

small_stmt     : expr_stmt
               | return_stmt
               | pass_stmt
               | break_stmt
               | continue_stmt
               | import_stmt
               | global_stmt
               | del_stmt
               | assert_stmt
               | raise_stmt

expr_stmt      : testlist                       /* sentencia de expresión: f(x)   */
               | testlist EQUAL assign_tail      /* asignación: a = b, a, b = ...  */

assign_tail    : testlist                        /* a = b            */
               | testlist EQUAL assign_tail      /* a = b = c (encadenada) */

return_stmt    : RETURN
               | RETURN testlist

pass_stmt      : PASS
break_stmt     : BREAK
continue_stmt  : CONTINUE

global_stmt    : GLOBAL name_list
del_stmt       : DEL testlist
assert_stmt    : ASSERT test
               | ASSERT test COMMA test
raise_stmt     : RAISE
               | RAISE test

name_list      : IDENTIFIER
               | name_list COMMA IDENTIFIER
```

> **Nota de diseño (asignación):** el lado izquierdo se reconoce como `testlist`
> general. Esto admite sintácticamente `n_rows, n_cols = x.shape` (desempaque) y
> también `a = b = c`. Validar que el blanco de asignación sea un objetivo válido
> (no, p. ej., `a + b = c`) es responsabilidad de la fase **semántica**, no del
> parser. Es una simplificación deliberada y estándar.

### 2.2 Sentencias compuestas y bloques

```
compound_stmt  : if_stmt
               | for_stmt
               | while_stmt
               | funcdef
               | decorated

suite          : simple_stmt                          /* cuerpo en una línea */
               | NEWLINE INDENT stmt_list DEDENT       /* bloque indentado    */

if_stmt        : IF test COLON suite elif_clauses else_clause

elif_clauses   : ε
               | elif_clauses ELIF test COLON suite

else_clause    : ε
               | ELSE COLON suite

for_stmt       : FOR testlist IN testlist COLON suite else_clause
while_stmt     : WHILE test COLON suite else_clause
```

> **Eliminación del dangling-else:** la forma de una línea de `suite` solo permite
> `simple_stmt` (un *small statement*), nunca otra sentencia compuesta. Por lo
> tanto un `if` anidado obliga a un bloque `INDENT … DEDENT` explícitamente
> delimitado, y el `else` siempre se asocia sin ambigüedad al `if` de su mismo
> nivel de sangría. La gramática es libre de dangling-else **por construcción**.

### 2.3 Definición de funciones y decoradores

```
funcdef        : DEF IDENTIFIER OPEN_PAREN param_list_opt CLOSE_PAREN COLON suite

decorated      : decorators funcdef

decorators     : decorator
               | decorators decorator

decorator      : AT_SIGN dotted_name NEWLINE
               | AT_SIGN dotted_name OPEN_PAREN arg_list_opt CLOSE_PAREN NEWLINE

param_list_opt : ε
               | param_list

param_list     : param
               | param_list COMMA param

param          : IDENTIFIER                          /* a                       */
               | IDENTIFIER COLON test               /* BLOCK_SIZE: tl.constexpr */
               | IDENTIFIER EQUAL test               /* a = 0                   */
               | IDENTIFIER COLON test EQUAL test    /* a: int = 0              */
               | MULTIPLY IDENTIFIER                  /* *args                   */
               | POWER IDENTIFIER                     /* **kwargs (** = POWER)   */

dotted_name    : IDENTIFIER
               | dotted_name DOT IDENTIFIER
```

> `@triton.jit` ⇒ `AT_SIGN dotted_name NEWLINE`.
> `@triton.autotune(configs=..., key=...)` ⇒ forma con llamada.
> `**kwargs` reutiliza el token `POWER` (`**`) y `*args` el token `MULTIPLY` (`*`).

### 2.4 Imports

```
import_stmt    : IMPORT dotted_as_names
               | FROM dotted_name IMPORT import_targets

dotted_as_names: dotted_as_name
               | dotted_as_names COMMA dotted_as_name

dotted_as_name : dotted_name
               | dotted_name AS IDENTIFIER

import_targets : MULTIPLY                                       /* from x import * */
               | import_as_names
               | OPEN_PAREN import_as_names CLOSE_PAREN

import_as_names: import_as_name
               | import_as_names COMMA import_as_name

import_as_name : IDENTIFIER
               | IDENTIFIER AS IDENTIFIER
```

> Cubre `import torch`, `import triton.language as tl`,
> `from torch import empty_like`, `from x import (a, b)`.

---

## 3. Expresiones (cascada de precedencia — inambigua)

De menor a mayor precedencia. Cada nivel es recursivo por la izquierda
(asociatividad izquierda), salvo `power` (derecha) y `factor` (unario, derecha).

```
test           : or_test

or_test        : and_test
               | or_test OR and_test

and_test       : not_test
               | and_test AND not_test

not_test       : NOT not_test
               | comparison

comparison     : bitor_expr
               | comparison comp_op bitor_expr

comp_op        : LESS_THAN | GREATER_THAN | DOUBLE_EQUAL | NOT_EQUAL
               | LESS_EQUAL | GREATER_EQUAL
               | IN | NOT IN | IS | IS NOT

bitor_expr     : bitxor_expr
               | bitor_expr BIT_OR bitxor_expr

bitxor_expr    : bitand_expr
               | bitxor_expr BIT_XOR bitand_expr

bitand_expr    : shift_expr
               | bitand_expr BIT_AND shift_expr

shift_expr     : arith_expr
               | shift_expr LEFT_SHIFT arith_expr
               | shift_expr RIGHT_SHIFT arith_expr

arith_expr     : term
               | arith_expr PLUS term
               | arith_expr MINUS term

term           : factor
               | term MULTIPLY factor
               | term DIVIDE factor
               | term MODULO factor
               | term FLOOR_DIV factor

factor         : PLUS factor                /* +x  unario */
               | MINUS factor               /* -x  unario */
               | BIT_NOT factor             /* ~x  unario */
               | power

power          : atom_expr
               | atom_expr POWER factor     /* ** asociativa por la derecha */
```

> **Por qué cascada y no `expr OP expr` + `%left`:** la cascada produce una CFG
> *inambigua sin* directivas de precedencia. Las producciones unitarias
> (`or_test → and_test → …`) **se conservan a propósito**: codifican la
> precedencia. Eliminarlas (como "simplificación" automática) reintroduciría la
> ambigüedad, así que NO se eliminan. Esta es la justificación que pide la rúbrica
> sobre producciones unitarias.

### 3.1 Postfijos: llamadas, indexado, atributos

```
atom_expr      : atom
               | atom_expr OPEN_PAREN arg_list_opt CLOSE_PAREN          /* f(...)    */
               | atom_expr OPEN_BRACKET subscript_list CLOSE_BRACKET    /* a[...]    */
               | atom_expr DOT IDENTIFIER                               /* a.b       */
```

> Maneja el *launch* de kernel `softmax_kernel[(n_rows,)](y, x, ...)` como
> `atom` (`softmax_kernel`) → subíndice `[(n_rows,)]` → llamada `(...)`.

### 3.2 Argumentos de llamada

```
arg_list_opt   : ε
               | arg_list

arg_list       : argument
               | arg_list COMMA argument

argument       : test                       /* posicional: input_ptrs        */
               | IDENTIFIER EQUAL test       /* por palabra clave: mask=mask  */
               | MULTIPLY test               /* *args                         */
               | POWER test                  /* **kwargs                      */
```

> Cubre `tl.load(input_ptrs, mask=mask, other=-float('inf'))` y
> `kernel[grid](y, x, n_cols, BLOCK_SIZE=BLOCK_SIZE)`.

### 3.3 Subíndices (incluye slice básico)

```
subscript_list : subscript_item
               | subscript_list COMMA subscript_item

subscript_item : test                        /* x[i] , x[i, j]   */
               | test_opt COLON test_opt     /* x[a:b] , x[:n]   */

test_opt       : ε
               | test
```

### 3.4 Átomos

```
atom           : IDENTIFIER
               | INT
               | FLOAT
               | SCIENTIFIC
               | STRING
               | TRUE
               | FALSE
               | NONE
               | OPEN_PAREN CLOSE_PAREN                  /* () tupla vacía         */
               | OPEN_PAREN testlist CLOSE_PAREN         /* (a)  grupo  /  (a,b) tupla */
               | OPEN_BRACKET CLOSE_BRACKET              /* []  lista vacía        */
               | OPEN_BRACKET testlist CLOSE_BRACKET     /* [a, b, c]  lista       */
               | OPEN_BRACE CLOSE_BRACE                  /* {} dict vacío          */
               | OPEN_BRACE dict_items CLOSE_BRACE       /* {k: v, ...}            */

testlist       : test
               | testlist COMMA test
               | testlist COMMA                          /* coma final: (n_rows,) */

dict_items     : dict_item
               | dict_items COMMA dict_item
               | dict_items COMMA

dict_item      : test COLON test
```

> La distinción "grupo `(a)`" vs "tupla `(a,)`/`(a, b)`" se decide por la presencia
> de coma; ambas comparten la misma regla y la diferencia es semántica.

---

## 4. Producciones epsilon y notas de simplificación

**Producciones epsilon (intencionales, todas justificadas):**
- `elif_clauses → ε`, `else_clause → ε` — cláusulas opcionales de `if`/`for`/`while`.
- `param_list_opt → ε`, `arg_list_opt → ε` — funciones/llamadas sin argumentos.
- `test_opt → ε` — extremos vacíos de un slice (`x[:n]`, `x[a:]`).

**Producciones unitarias:** las de la cascada de expresiones (Sección 3) se
conservan porque codifican precedencia/asociatividad. No son "inútiles": cada
nivel define un punto de entrada distinto para un operador de distinta precedencia.

**Símbolos inútiles:** todo no terminal es alcanzable desde `program` y deriva al
menos una cadena de terminales (no hay símbolos estériles ni inalcanzables).

---

## 5. Cobertura objetivo (kernel softmax de ejemplo)

La gramática reconoce, entre otros, todos los constructos del kernel softmax:

```python
import triton
import triton.language as tl          # FROM/IMPORT + AS

@triton.jit                            # decorator (dotted_name)
def softmax_kernel(output_ptr, input_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)         # asignación + llamada + atributo
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols        # comparación
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))  # kwargs + unario + STRING
    num = tl.exp(row - tl.max(row, axis=0))                    # aritmética anidada
    tl.store(output_ptrs, num / denom, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape           # desempaque de tupla
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](y, x, n_cols, BLOCK_SIZE=BLOCK_SIZE)  # launch
    return y
```

---

## 6. Pendientes / extensiones (fuera del alcance moderado, documentadas)

| Constructo | Por qué se excluye | Cómo añadirlo |
|---|---|---|
| Asignación aumentada `+= -= *= …` | Requiere tokens compuestos nuevos en el lexer | Añadir tokens y una regla `augassign` en `expr_stmt` |
| Anotación de retorno `-> tipo` | No existe token `->` (sería `MINUS GREATER_THAN`) | Añadir token `ARROW` al lexer |
| Expresión ternaria `a if c else b` | Complica la cascada | Añadir nivel sobre `or_test` |
| Comprehensions / `lambda` / `with` | Poco frecuentes en kernels | Reglas dedicadas |
| `try/except` detallado | Raro en kernels; tokens existen | `try_stmt` con cláusulas `except` |
