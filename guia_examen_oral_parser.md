# Guía de estudio — Analizador Sintáctico Triton (Fase II)

**Equipo 13 · TC3002B Gpo 501 · Preparación para el examen oral**

Esta guía te lleva desde cero hasta poder defender cada línea del proyecto. Está
organizada para que la leas en orden, pero también para que la uses como
referencia rápida el día del examen. Lo más importante para el oral está en las
secciones 4 (decisiones de diseño) y 8 (banco de preguntas).

## Índice
1. Qué es este proyecto y dónde encaja
2. Conceptos fundamentales (léxico, sintáctico, gramáticas)
3. Cómo funcionan flex y bison juntos
4. Las decisiones de diseño que TIENES que poder defender
5. Recorrido del código (qué hace cada parte y por qué)
6. Cómo fluye una ejecución (traza paso a paso)
7. Cómo compilar, correr y demostrar
8. Banco de preguntas del oral con respuestas modelo
9. Glosario rápido
10. Checklist del día del examen

---

## 1. Qué es este proyecto y dónde encaja

Un **compilador** traduce código fuente a otra representación (código máquina,
bytecode, etc.). Lo hace por etapas. Las dos primeras forman el *front-end*:

1. **Análisis léxico (Fase I, ya entregada):** lee los caracteres del archivo y
   los agrupa en **tokens** (palabras del lenguaje): identificadores, números,
   palabras reservadas, operadores. Es como separar una oración en palabras.
2. **Análisis sintáctico (Fase II, este proyecto):** toma esa secuencia de tokens
   y verifica que respeten la **estructura gramatical** del lenguaje. Es como
   verificar que las palabras formen oraciones válidas (sujeto, verbo, etc.).

El lenguaje que analizamos es un **subconjunto de Triton**, que es Python con
extensiones para escribir kernels de GPU. Nuestro parser reconoce imports,
definiciones de funciones decoradas (`@triton.jit`), control de flujo, asignaciones,
y expresiones con todos los operadores.

**La frase de una línea para el oral:** *"Construí un analizador sintáctico que,
sobre el flujo de tokens de la Fase I, verifica que un kernel Triton respete una
gramática libre de contexto inambigua, reporta errores de sintaxis con su línea, y
recoge información semántica (funciones y sus parámetros) durante el análisis."*

---

## 2. Conceptos fundamentales

### 2.1 Token, lexema y patrón
- **Lexema:** el texto concreto, p. ej. `softmax_kernel`.
- **Token:** la categoría a la que pertenece, p. ej. `IDENTIFIER`.
- **Patrón:** la regla (expresión regular) que define qué lexemas son de ese token,
  p. ej. una letra seguida de letras o dígitos.

El analizador léxico convierte lexemas en tokens. El parser solo trabaja con
tokens (categorías), no con el texto.

### 2.2 Gramática Libre de Contexto (CFG)
Una CFG describe la estructura de un lenguaje con cuatro componentes:
- **Terminales:** los tokens (las "hojas"), p. ej. `IF`, `IDENTIFIER`, `PLUS`.
- **No terminales:** categorías estructurales (las "ramas"), p. ej. `if_stmt`,
  `expr_stmt`, `test`.
- **Producciones (reglas):** cómo un no terminal se compone de otros símbolos,
  p. ej. `if_stmt : IF test COLON suite ...`.
- **Símbolo inicial:** el no terminal raíz; en nuestro caso `program`.

"Libre de contexto" significa que una producción se aplica a un no terminal sin
importar qué hay a su alrededor (el lado izquierdo es siempre un solo no terminal).

### 2.3 Derivación y árbol sintáctico
Una **derivación** es la secuencia de aplicaciones de reglas que parte del símbolo
inicial y llega a la cadena de tokens. El **árbol sintáctico** (parse tree) es la
representación en árbol de esa derivación: la raíz es `program`, las hojas son los
tokens. El parser, en el fondo, construye (implícitamente) ese árbol.

### 2.4 Ambigüedad
Una gramática es **ambigua** si una misma cadena de tokens tiene más de un árbol
sintáctico válido. La ambigüedad es mala porque el significado del programa quedaría
indefinido. **Un objetivo central de este proyecto es que la gramática sea
inambigua.** Dos fuentes clásicas de ambigüedad que tuvimos que resolver:
precedencia de operadores y el *dangling-else* (sección 4).

### 2.5 Parsers descendentes vs ascendentes (LL vs LR/LALR)
- **Descendente (LL):** construye el árbol de la raíz hacia las hojas; predice qué
  regla usar viendo los siguientes tokens. Se puede escribir a mano (descenso
  recursivo).
- **Ascendente (LR / LALR):** construye el árbol de las hojas hacia la raíz;
  va **apilando** tokens (shift) y, cuando reconoce el lado derecho de una regla,
  lo **reduce** al no terminal (reduce). Es más potente. **bison/yacc genera un
  parser LALR(1)**: ascendente, con 1 token de anticipación (lookahead).

### 2.6 Shift / Reduce y conflictos
El parser LALR tiene una pila y, en cada paso, decide:
- **Shift:** meter el siguiente token a la pila.
- **Reduce:** reconocer que el tope de la pila es el lado derecho de una regla y
  sustituirlo por el lado izquierdo.

Un **conflicto shift/reduce** ocurre cuando, con la información disponible, el parser
no sabe si debe hacer shift o reduce. Un **conflicto reduce/reduce** es cuando no
sabe por cuál de dos reglas reducir. Una gramática inambigua y bien diseñada produce
**0 conflictos** (lo nuestro: bison reporta 0 en `parser.output`).

---

## 3. Cómo funcionan flex y bison juntos

Estas son dos herramientas clásicas de Unix que se complementan:

- **flex** genera el **scanner** (analizador léxico) a partir de reglas de
  expresiones regulares. Internamente construye un **autómata finito** que reconoce
  los patrones. La función generada es `yylex()`, que devuelve **un token por
  llamada**.
- **bison** (versión GNU de yacc) genera el **parser** a partir de la gramática.
  La función generada es `yyparse()`, que pide tokens llamando a `yylex()` hasta
  aceptar o encontrar un error.

### El puente entre ambos
1. `bison -d parser.y` genera `parser.tab.c` (el parser) **y** `parser.tab.h` (un
   header con las **macros de los tokens**: `#define IDENTIFIER 258`, etc.).
2. El lexer hace `#include "parser.tab.h"` y, en cada regla, hace
   `return IDENTIFIER;` (la macro), devolviendo ese número al parser.
3. Para tokens que llevan valor (como el texto de un identificador), el lexer
   escribe en la variable global compartida **`yylval`** antes de hacer `return`.
   El parser lee ese valor con `$1`, `$2`, etc.
4. `gcc parser.tab.c lex.yy.c -o triton_parser` enlaza ambos en un ejecutable.

**Diagrama mental del flujo de datos:**
```
archivo.triton
      │
      ▼
  flex (yylex)  ──tokens──►  bison (yyparse)  ──►  ¿válido? + errores + tablas
   lex.yy.c                    parser.tab.c
```

### Punto fino importante (suele preguntarse)
Hay **dos numeraciones de tokens distintas** en el proyecto:
- Los **IDs 1–62** del reporte de la Fase I: un catálogo nuestro para el JSON léxico.
- Los **códigos de `parser.tab.h`** (desde 258): los que bison asigna y el parser
  usa de verdad.
El lexer ahora devuelve los **segundos** (`return IDENTIFIER;`, no `28`). No se
contradicen: son dos cosas para dos propósitos.

---

## 4. Las decisiones de diseño que TIENES que poder defender

Esta es la sección clave del oral. Para cada decisión: qué hicimos, por qué, y qué
pasaría si no.

### 4.1 Tres tokens estructurales nuevos: `NEWLINE`, `INDENT`, `DEDENT`
**Qué:** agregamos al scanner tres tokens que no existían en la Fase I.
**Por qué:** Triton usa la sintaxis de Python, donde **la estructura de bloques se
expresa con la sangría** (indentación), no con llaves. El lexer de la Fase I
descartaba los saltos de línea y los espacios, así que el flujo de tokens no podía
expresar dónde empieza ni termina un bloque.
- `NEWLINE` marca el fin de una **línea lógica** (una sentencia).
- `INDENT` se emite cuando la sangría **aumenta** (se abre un bloque).
- `DEDENT` se emite cuando la sangría **disminuye** (se cierra un bloque).
**Cómo:** una **pila de indentación**. Cada línea mide su sangría y la compara con
el tope de la pila: si es mayor, push + `INDENT`; si es menor, pop(s) + `DEDENT`(s)
hasta igualar; si es igual, nada.
**Si no:** sería imposible distinguir el cuerpo de un `if` del código que le sigue.
Esto es exactamente la modificación al scanner que permite la rúbrica (Paso 3).

### 4.2 La regla `suite` y la eliminación del *dangling-else*
**El problema clásico (dangling-else):** en gramáticas tipo C, `if a then if b then
s1 else s2` es ambiguo: ¿el `else` es del `if` interno o del externo?
**Nuestra solución:** la regla del cuerpo de un bloque es
```
suite : simple_stmt                       (una sola línea)
      | NEWLINE INDENT stmt_list DEDENT    (bloque indentado)
```
y la forma de **una sola línea** solo admite `simple_stmt` (una sentencia *simple*,
nunca compuesta). Por lo tanto **un `if` anidado obliga a un bloque indentado**, que
está delimitado explícitamente por `INDENT`/`DEDENT`. Así, el `else` siempre se
asocia sin ambigüedad al `if` de su mismo nivel de sangría.
**Frase para el oral:** *"Mi gramática es libre de dangling-else por construcción:
los bloques están delimitados por INDENT/DEDENT, así que el else nunca es ambiguo."*

### 4.3 Cascada de precedencia (expresiones inambiguas sin directivas)
**Qué:** en vez de escribir `expr : expr OP expr` y resolver con directivas de
precedencia de yacc (`%left`, `%right`), escribimos una **cascada** de no terminales,
uno por nivel de precedencia:
```
test → or_test → and_test → not_test → comparison → bitor_expr → bitxor_expr →
bitand_expr → shift_expr → arith_expr → term → factor → power → atom_expr → atom
```
**Por qué:** la cascada hace la gramática **inambigua por construcción**. Cada nivel
solo puede combinarse con el nivel inmediatamente superior, lo que fija precedencia
y asociatividad estructuralmente. Es lo que pide la rúbrica (gramática inambigua) sin
depender de un mecanismo externo del generador.
**Asociatividad:**
- **Izquierda** (la mayoría): se logra con **recursión por la izquierda**, p. ej.
  `arith_expr : arith_expr PLUS term`. Así `a - b - c` se agrupa `(a - b) - c`.
- **Derecha** (potencia `**`): `power : atom_expr POWER factor`, donde el operando
  derecho es `factor` (que vuelve a contener `power`). Así `2 ** 3 ** 2` se agrupa
  `2 ** (3 ** 2)`, como en matemáticas y en Python.
**Producciones unitarias (clave para el oral):** las reglas como
`or_test : and_test` son producciones unitarias. **No se eliminan a propósito**:
codifican la precedencia. Si las "simplificáramos", reintroduciríamos la ambigüedad.
Es la justificación que pide la rúbrica sobre simplificaciones.

### 4.4 El objetivo del `for` usa `exprlist` (no `testlist`)
**El conflicto:** el token `IN` aparece en dos lugares: como palabra clave del `for`
(`for i in xs`) y como operador de comparación (`a in b`). Si el objetivo del `for`
fuera una expresión general (`testlist`, que incluye comparaciones), al leer
`for i in xs` el parser podría interpretar `i in xs` como **una sola comparación** y
luego no encontraría el `IN` del bucle.
**La solución:** el objetivo del `for` usa `exprlist`, una lista basada en
`bitor_expr`, que está **un nivel por debajo** de `comparison` en la cascada. Como
`exprlist` no puede contener `in`, el primer `IN` que aparece es inequívocamente el
del bucle. (Es el mismo diseño que usa la gramática oficial de Python con `exprlist`.)

### 4.5 El lado izquierdo de una asignación es un `testlist`
**Qué:** `expr_stmt : testlist | testlist EQUAL assign_tail`. El lado izquierdo se
reconoce como una expresión general.
**Por qué:** así admitimos desempaque de tuplas (`n_rows, n_cols = x.shape`) y
asignación encadenada (`a = b = c`) con una sola regla limpia.
**El matiz:** sintácticamente esto también aceptaría algo como `a + b = c`, que es
inválido. **Validar que el destino sea asignable es tarea de la fase semántica, no
del parser.** Separar sintaxis de semántica es una decisión de diseño correcta y
estándar (lo hace el propio CPython).

### 4.6 Atributos sintetizados para contar parámetros
**Qué:** la "actualización semántica" del parser es registrar cada función y su
número de parámetros al reducir `funcdef`. El conteo se hace con **atributos
sintetizados**:
```
param_list_opt : /* vacío */   { $$ = 0; }
               | param_list     { $$ = $1; }
param_list     : param                  { $$ = 1; }
               | param_list COMMA param  { $$ = $1 + 1; }
funcdef        : DEF IDENTIFIER OPEN_PAREN param_list_opt CLOSE_PAREN COLON suite
                   { register_function($2, $4); }
```
**Conceptos:** `$$` es el valor que produce la regla (sintetizado de abajo hacia
arriba). `$1`, `$2`, `$4` son los valores de los símbolos del lado derecho (el 1º,
2º, 4º). Aquí `$2` es el texto del nombre (lo trae el lexer en `yylval.lexeme`) y
`$4` es el conteo de parámetros propagado por `param_list`.
**Por qué importa:** el parser puede clasificar identificadores por su **rol**
(nombre de función, parámetro) porque conoce el **contexto sintáctico**; el lexer
solo no puede.

### 4.7 Recuperación de errores (`error NEWLINE` + `yyerrok`)
**Qué:** `statement : error NEWLINE { yyerrok; }`.
**Por qué:** sin recuperación, el parser se detiene en el primer error. Con la regla
`error`, bison descarta tokens hasta encontrar un `NEWLINE`, **re-sincroniza al inicio
de la siguiente línea** y continúa, lo que permite **reportar varios errores en una
sola corrida**. `yyerrok` le dice a bison que ya se recuperó y puede volver a modo
normal. `error` es un token especial reservado de bison precisamente para esto.

### 4.8 Wrapper `yylex` / `raw_lex` y la cola de `DEDENT`s
**El problema:** `yyparse` pide **un token por llamada** a `yylex`. Pero al cerrar
varios bloques a la vez (p. ej. salir de un `for` dentro de un `if`), hay que emitir
**varios `DEDENT` seguidos**. Una sola llamada no puede devolver varios tokens.
**La solución:** renombramos el scanner de flex a `raw_lex` (con `YY_DECL`) y
escribimos a mano un `yylex()` que envuelve a `raw_lex`. Mantenemos una **cola** de
tokens estructurales pendientes; `yylex` la drena uno por uno antes de volver a
llamar a `raw_lex`. Cuando una línea necesita 3 `DEDENT`, se encolan los 3 y se
entregan en 3 llamadas.

### 4.9 El bug del *match vacío* en flex (y cómo lo arreglamos)
**Esto demuestra entendimiento real; vale oro si te preguntan por dificultades.**
La medición de sangría se hace en un estado de inicio de línea (`%x BOL`). El primer
intento usaba la regla `<BOL>[ \t]*` para medir la sangría. Problema: en una línea
**sin sangría**, esa regla hace *match de la cadena vacía* (0 caracteres), y **flex
no ejecuta la acción de un match vacío**: en su lugar hace ECHO del carácter y se
queda en el estado, descomponiendo el flujo de tokens.
**El arreglo:** la regla ahora es `<BOL>[ \t]*[^ \t\r\n#]`, que matchea los espacios
de sangría **más el primer carácter real** (así el match siempre tiene ≥1 carácter y
flex sí dispara la acción). Medimos la sangría (la función solo cuenta espacios/tabs,
así que el carácter real no afecta) y luego devolvemos ese carácter al flujo con
`yyless(yyleng - 1)` para que las reglas normales lo re-escaneen.
**Frase para el oral:** *"Descubrí que flex no dispara acciones en matches vacíos, así
que reformulé la regla de sangría para que consuma al menos un carácter y lo regreso
al flujo con yyless."*

### 4.10 Simplificaciones documentadas (lo que la rúbrica pide justificar)
- **Producciones epsilon (vacías):** cláusulas opcionales (`elif_clauses`,
  `else_clause`, `param_list_opt`, `arg_list_opt`, `test_opt` para slices). Son
  intencionales y necesarias para expresar "0 o más" / "opcional".
- **Producciones unitarias:** las de la cascada de expresiones; se conservan porque
  codifican precedencia (ver 4.3).
- **Símbolos inútiles:** no hay. Todo no terminal es alcanzable desde `program` y
  deriva al menos una cadena de terminales.

### 4.11 Alcance (qué dejamos fuera y por qué)
Elegimos un **subconjunto moderado y realista**. Excluimos, documentándolo:
- **Asignación aumentada (`+=`, `-=`, ...):** requiere tokens compuestos nuevos en el
  lexer. Se puede añadir con una regla `augassign`.
- **Anotación de retorno `-> tipo`:** no existe el token `->` (sería `MINUS`
  `GREATER_THAN`).
- **Ternario, comprehensions, `lambda`, `with`, `try/except` detallado:** poco
  frecuentes en kernels.
Justificar el alcance es parte de la rúbrica: muestra criterio de ingeniería.

---

## 5. Recorrido del código

### 5.1 `parser.y` — estructura de un archivo bison
Un archivo `.y` tiene tres secciones separadas por `%%`:
```
%{  ... código C (includes, variables, funciones auxiliares) ... %}
... declaraciones bison (%union, %token, %type, %start) ...
%%
... reglas gramaticales con acciones { ... } ...
%%
... código C de usuario (yyerror, main) ...
```
Elementos clave:
- **`%union { int entry; char *lexeme; }`**: define los tipos posibles del valor
  semántico de cada símbolo. Un token/no terminal usa uno de estos campos.
- **`%token <lexeme> IDENTIFIER`**: declara el token `IDENTIFIER` y dice que su valor
  vive en el campo `lexeme` de la unión.
- **`%type <entry> param_list param_list_opt`**: declara que esos no terminales
  producen un valor en el campo `entry` (el conteo de parámetros).
- **`%start program`**: el símbolo inicial.
- **Acciones semánticas** `{ ... }`: código C que corre **al reducir** esa regla.
  Dentro se usan `$$`, `$1`, `$2`, ...

Funciones del epílogo:
- **`yyerror(const char *s)`**: la llama bison al detectar un error; nosotros
  incrementamos el contador y reportamos a `stderr` con `yylineno`.
- **`main()`**: abre el archivo (o lee de `stdin`), llama a `yyparse()`, y al final
  imprime el resultado, las funciones reconocidas y las tablas de símbolos.

### 5.2 `lexicalAnalyzer.l` — estructura de un archivo flex
Mismo esquema de tres secciones. Elementos clave:
- **`%option noyywrap yylineno`**: `noyywrap` evita necesitar la función `yywrap`;
  `yylineno` activa el conteo automático de líneas (lo usa `yyerror`).
- **`%x BOL`**: declara un **estado de inicio exclusivo** llamado `BOL` (Beginning Of
  Line). En un estado exclusivo solo están activas las reglas marcadas con `<BOL>`.
- **`#define YY_DECL int raw_lex(void)`**: renombra la función generada por flex a
  `raw_lex` (para envolverla con nuestro `yylex`).
- **Definiciones de patrones** (`ID`, `SCI`, `FLOAT`, `INT`, `STR_DQ`, `STR_SQ`).
- **Reglas:** los operadores compuestos (`**`, `==`, `<=`, ...) van **antes** que sus
  prefijos para que gane el match más largo (*maximal munch*).
- **El macro `RET(tok)`**: marca que la línea tuvo tokens (`line_has_tokens = 1`) y
  hace `return tok`.
- **Funciones auxiliares:** `add_identifier`, `add_number`, `add_string` (tablas de
  símbolos), `handle_indentation` (pila de sangría), `q_push`/`q_pop` (cola),
  `indent_width` (mide la sangría), y el wrapper `yylex`.

### 5.3 Por qué el orden de las reglas importa (maximal munch)
flex elige **el match más largo**; si dos reglas matchean lo mismo de largo, gana
**la que aparece primero**. Por eso:
- `"**"` antes que `"*"` (si no, `**` se leería como dos `*`).
- Palabras reservadas (`"def"`) antes que `{ID}` (si no, `def` sería un identificador).
  Para `define`, `{ID}` matchea 6 caracteres y `"def"` solo 3, así que gana `{ID}`:
  correcto.
- `{SCI}` antes que `{FLOAT}` antes que `{INT}` (un número científico es el match
  más específico/largo).

---

## 6. Cómo fluye una ejecución (traza paso a paso)

Tomemos esta entrada mínima:
```
def f():
    return 1
```

**Flujo de tokens que produce el lexer** (lo puedes mostrar en el oral):
```
DEF  IDENTIFIER(f)  OPEN_PAREN  CLOSE_PAREN  COLON  NEWLINE
INDENT  RETURN  INT(1)  NEWLINE  DEDENT  <EOF>
```
Paso a paso del lexer:
1. Estado `BOL`, línea `def f():`. Mide sangría 0 (igual al tope) → no emite nada,
   pasa a estado normal.
2. `def`→`DEF`, `f`→`IDENTIFIER`, `(`→`OPEN_PAREN`, `)`→`CLOSE_PAREN`, `:`→`COLON`.
3. `\n` → la línea tuvo tokens → emite `NEWLINE`, vuelve a `BOL`.
4. `BOL`, línea `    return 1`. Mide sangría 4 > 0 → push, emite `INDENT`.
5. `return`→`RETURN`, `1`→`INT`.
6. `\n` → emite `NEWLINE`, vuelve a `BOL`.
7. `BOL`, fin de archivo. `raw_lex` devuelve 0. El wrapper `yylex` cierra: la última
   línea no tenía token pendiente, pero queda un nivel de sangría abierto → emite
   `DEDENT`. Luego `<EOF>`.

**Cómo lo reduce el parser** (de abajo hacia arriba):
- `INT(1)` se reduce subiendo por la cascada hasta `testlist`; con `RETURN testlist`
  forma un `return_stmt`; con el `NEWLINE` forma un `simple_stmt`.
- `NEWLINE INDENT [ese simple_stmt] DEDENT` reduce a `suite`.
- `DEF IDENTIFIER OPEN_PAREN param_list_opt(=0) CLOSE_PAREN COLON suite` reduce a
  `funcdef`, y se dispara la acción `register_function("f", 0)`.
- `funcdef` → `compound_stmt` → `statement` → `stmt_list` → `program`. **Aceptado.**

---

## 7. Cómo compilar, correr y demostrar

```
make            # bison -d -v parser.y ; flex lexicalAnalyzer.l ; gcc ... -o triton_parser
make run        # corre sobre example.triton
./triton_parser example.triton          # caso válido (softmax)
./triton_parser example_control.triton  # if/elif/else, for, while
make clean
```
**Para demostrar que detecta errores**, crea un archivo con `y = = 7` o un paréntesis
sin cerrar y córrelo: verás `Error de sintaxis: ... (linea N)` y que continúa.

**Para demostrar 0 conflictos**, abre `parser.output` (lo genera `bison -v`) y busca
la palabra "conflict": no debe aparecer ninguna.

Resultados ya verificados:
- `example.triton` → "Analisis sintactico exitoso", 2 funciones (`softmax_kernel`/6,
  `softmax`/1), 43 identificadores, 1 número (`0`), 1 string (`inf`).
- `example_control.triton` → válido, `add_kernel`/5 parámetros.
- archivo con errores → reporta varios errores (sintácticos y léxico) y se recupera.

---

## 8. Banco de preguntas del oral con respuestas modelo

**P: ¿Cuál es la diferencia entre análisis léxico y sintáctico?**
R: El léxico agrupa caracteres en tokens (categorías); el sintáctico verifica que esa
secuencia de tokens respete la estructura gramatical del lenguaje. El léxico no sabe
de estructura; el sintáctico no vuelve a mirar caracteres, solo tokens.

**P: ¿Qué tipo de parser genera bison?**
R: Un parser **LALR(1)**: ascendente (de hojas a raíz), basado en una pila, con
acciones shift/reduce y 1 token de anticipación.

**P: ¿Qué es un conflicto shift/reduce? ¿Tuviste alguno?**
R: Es cuando el parser no puede decidir entre apilar el siguiente token o reducir el
tope de la pila. Mi gramática produce **0 conflictos**, lo cual verifico en
`parser.output`. Lo logré con una cascada de precedencia inambigua y restringiendo
el objetivo del `for`.

**P: ¿Por qué tu gramática es inambigua?**
R: Por dos diseños: (1) la precedencia y asociatividad de operadores están
codificadas en una **cascada** de no terminales, así que cada expresión tiene un
único árbol; (2) el *dangling-else* se elimina porque los bloques están delimitados
por `INDENT`/`DEDENT` y el cuerpo de una línea solo admite sentencias simples.

**P: ¿Cómo resolviste el dangling-else?**
R: Por construcción. La regla `suite` de una sola línea solo permite `simple_stmt`,
nunca una sentencia compuesta. Entonces un `if` anidado obliga a un bloque indentado
delimitado por `INDENT`/`DEDENT`, y el `else` siempre se asocia sin ambigüedad.

**P: ¿Por qué `**` (potencia) es asociativo por la derecha?**
R: Porque en `power : atom_expr POWER factor`, el operando derecho es `factor`, que
vuelve a contener `power`. Así `2 ** 3 ** 2` se agrupa `2 ** (3 ** 2)`, igual que en
matemáticas y en Python.

**P: ¿Cómo logras asociatividad por la izquierda en la suma?**
R: Con recursión por la izquierda: `arith_expr : arith_expr PLUS term`. Eso agrupa
`a - b - c` como `(a - b) - c`.

**P: ¿Qué tokens nuevos agregaste y por qué?**
R: `NEWLINE`, `INDENT` y `DEDENT`. Triton usa sintaxis de Python, donde los bloques se
definen por sangría. El lexer original descartaba saltos de línea y espacios, así que
el flujo de tokens no podía expresar la estructura de bloques. Los genero con una pila
de indentación.

**P: ¿Cómo funciona tu manejo de indentación?**
R: En el inicio de cada línea mido la sangría y la comparo con el tope de una pila: si
es mayor, hago push y emito `INDENT`; si es menor, hago pop(s) y emito un `DEDENT` por
cada nivel cerrado; si es igual, no emito nada. Las líneas en blanco y de comentario
se ignoran (no cambian la sangría).

**P: Tu `yylex` no es el de flex. ¿Por qué?**
R: Porque al cerrar varios bloques a la vez hay que emitir varios `DEDENT`, pero
`yyparse` pide un token por llamada. Renombré el scanner de flex a `raw_lex` y escribí
un `yylex` que mantiene una cola: drena los tokens estructurales pendientes uno por uno
antes de volver a llamar a `raw_lex`. También al EOF emite el `NEWLINE` final y vacía
los `DEDENT` que falten.

**P: ¿Cómo se comunican flex y bison?**
R: `bison -d` genera `parser.tab.h` con las macros de los tokens. El lexer lo incluye
y hace `return` de esas macros. Los valores asociados (como el texto de un
identificador) viajan en la variable global `yylval`. `yyparse` llama a `yylex` para
pedir tokens.

**P: ¿Qué es `yylval`?**
R: La variable global donde el lexer deja el **valor semántico** del token actual (su
tipo lo define `%union`). Para `IDENTIFIER` guardo el lexema en `yylval.lexeme`, y el
parser lo lee con `$1`, `$2`, etc.

**P: Explícame esta acción: `register_function($2, $4);`.**
R: Está en la regla de `funcdef`. `$2` es el `IDENTIFIER` (el nombre de la función),
`$4` es el número de parámetros que calculé con atributos sintetizados en
`param_list`. Al reducir la función completa, registro su nombre y su aridad: es la
actualización semántica que recoge el parser.

**P: ¿Qué es un atributo sintetizado?**
R: Un valor que un no terminal **produce** (en `$$`) a partir de los valores de sus
hijos, de abajo hacia arriba. Mi conteo de parámetros es sintetizado:
`param_list COMMA param { $$ = $1 + 1; }`.

**P: ¿Por qué el objetivo del `for` no es una expresión general?**
R: Porque el token `IN` es a la vez palabra clave del `for` y operador de comparación.
Si el objetivo admitiera comparaciones, `for i in xs` sería ambiguo. Uso `exprlist`,
que está por debajo del nivel de comparación, así que no puede contener `in` y el
primer `IN` es siempre el del bucle.

**P: ¿Tu parser acepta `a + b = c`? ¿No es un error?**
R: Sintácticamente sí lo acepta, porque el lado izquierdo es una expresión general
para permitir desempaque de tuplas y asignación encadenada. Verificar que el destino
sea asignable es responsabilidad de la **fase semántica**, no del análisis sintáctico.
Separar ambas cosas es estándar (lo hace el propio Python).

**P: ¿Qué pasa cuando hay un error de sintaxis?**
R: bison llama a `yyerror`, que reporta el mensaje y la línea. Gracias a la regla
`statement : error NEWLINE { yyerrok; }`, descarto tokens hasta el siguiente
`NEWLINE`, me re-sincronizo y continúo, de modo que reporto varios errores en una sola
corrida.

**P: ¿Qué son las producciones unitarias y por qué no las eliminaste?**
R: Son reglas como `or_test : and_test`, donde un no terminal deriva a otro. En mi
cascada codifican la precedencia de operadores; si las eliminara, reintroduciría
ambigüedad. Por eso se conservan a propósito.

**P: ¿Qué son las producciones epsilon en tu gramática?**
R: Producciones vacías para cosas opcionales: las cláusulas `elif`/`else`, las listas
de parámetros y argumentos vacías, y los extremos vacíos de un slice. Son necesarias
para expresar "opcional" o "cero o más".

**P: ¿Hay símbolos inútiles en tu gramática?**
R: No. Todo no terminal es alcanzable desde `program` y deriva al menos una cadena de
terminales.

**P: ¿Por qué usaste yacc en vez de escribir el parser a mano?**
R: Porque la rúbrica lo pide y porque yacc genera un parser LALR(1) correcto y
eficiente a partir de la gramática, con manejo de la pila y detección de conflictos.
Escribirlo a mano sería más propenso a errores y más difícil de mantener.

**P: ¿Qué dificultad técnica encontraste?**
R: La medición de sangría. Descubrí que flex **no ejecuta la acción de un match
vacío**: en una línea sin sangría mi regla `[ \t]*` matcheaba la cadena vacía y flex
hacía ECHO en lugar de procesarla. Lo arreglé haciendo que la regla consuma también el
primer carácter real (`[ \t]*[^ \t\r\n#]`) y devolviéndolo al flujo con `yyless`.

**P: ¿Por qué los códigos de token no son los IDs 1–62 del reporte léxico?**
R: Son dos numeraciones para dos propósitos. Los 1–62 eran un catálogo nuestro para el
JSON de la Fase I. Los códigos reales que usa el parser los asigna bison (desde 258) y
viven en `parser.tab.h`. El lexer ahora devuelve estos últimos.

**P: ¿Cómo manejas comentarios y líneas en blanco?**
R: Se ignoran por completo: no producen `NEWLINE` ni cambian la indentación, igual que
en Python. Los detecto en el estado de inicio de línea antes de medir la sangría.

---

## 9. Glosario rápido
- **Token / lexema / patrón:** categoría / texto concreto / regla que lo define.
- **CFG:** gramática libre de contexto (terminales, no terminales, producciones,
  símbolo inicial).
- **Ambigüedad:** una cadena con más de un árbol sintáctico.
- **LALR(1):** parser ascendente con 1 token de anticipación (lo que genera bison).
- **Shift / reduce:** apilar el siguiente token / reconocer y colapsar una regla.
- **Conflicto:** el parser no puede decidir entre shift y reduce (o entre dos reduces).
- **Atributo sintetizado:** valor que un nodo produce a partir de sus hijos (`$$`).
- **Producción epsilon:** producción vacía (opcional / cero o más).
- **Producción unitaria:** un no terminal deriva a otro solo (`A : B`).
- **Dangling-else:** ambigüedad sobre a qué `if` pertenece un `else`.
- **Maximal munch:** flex prefiere el match más largo; a igual longitud, la primera
  regla.
- **`yylex` / `yyparse` / `yylval` / `yyerror` / `yylineno`:** scanner / parser /
  valor del token / manejador de error / línea actual.
- **`yyless(n)`:** conserva `n` caracteres del match y devuelve el resto al flujo.
- **Estado de inicio (`%x`):** modo exclusivo del scanner; solo sus reglas activas.

---

## 10. Checklist del día del examen
- [ ] Sé explicar, sin leer, qué hace el proyecto en una frase.
- [ ] Puedo dibujar el flujo: `.triton` → flex → bison → resultado.
- [ ] Puedo nombrar y justificar los 3 tokens nuevos y la pila de indentación.
- [ ] Puedo explicar por qué la gramática es inambigua (cascada + dangling-else).
- [ ] Puedo explicar la asociatividad de `+` (izquierda) y `**` (derecha).
- [ ] Puedo explicar `exprlist` en el `for` y el conflicto del `IN`.
- [ ] Puedo leer una acción semántica y decir qué hace cada `$n`.
- [ ] Puedo explicar la recuperación de errores y demostrarla en vivo.
- [ ] Puedo explicar el wrapper `yylex` y la cola de `DEDENT`.
- [ ] Puedo contar el bug del match vacío y su arreglo con `yyless`.
- [ ] Puedo compilar con `make` y correr los tres casos (válido, control, errores).
- [ ] Puedo abrir `parser.output` y mostrar 0 conflictos.
- [ ] Sé qué dejé fuera del alcance y por qué.
