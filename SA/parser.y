/* ==========================================================================
 *  parser.y  -  Analizador Sintáctico (Parser) para el lenguaje Triton.
 *  Equipo 13 - TC3002B Gpo 501 - Fase II: Análisis Sintáctico.
 *
 *  PROPÓSITO
 *  ---------
 *  Especificación yacc/bison de una Gramática Libre de Contexto (CFG)
 *  inambigua para un subconjunto moderado y realista de kernels Triton.
 *  El parser consume el flujo de tokens producido por el scanner
 *  (lexicalAnalyzer.l, ya modificado para hacer `return` de códigos de
 *  token), verifica que el programa respeta la estructura gramatical y
 *  reporta los errores de sintaxis encontrados.
 *
 *  RELACIÓN CON OTROS ARCHIVOS
 *  ---------------------------
 *    - lexicalAnalyzer.l : provee yylex(), yylineno e yyin, y devuelve los
 *                          códigos de token declarados aquí con %token
 *                          (exportados a y.tab.h). Para IDENTIFIER coloca el
 *                          lexema en yylval.lexeme.
 *    - y.tab.h           : generado por bison a partir de este archivo;
 *                          define las macros de token que usa el scanner.
 *
 *  CONSTRUCCIÓN
 *  ------------
 *    bison -d -v parser.y          # genera parser.tab.c, parser.tab.h, parser.output
 *    flex      lexicalAnalyzer.l   # genera lex.yy.c
 *    gcc parser.tab.c lex.yy.c -o triton_parser
 *
 *  Revisar parser.output para confirmar 0 conflictos shift/reduce.
 * ========================================================================== */

%{
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    /* --- Interfaz provista por el scanner (flex) --- */
    int  yylex(void);            /* siguiente token                          */
    extern int   yylineno;       /* línea actual (activada con %option yylineno) */
    extern FILE *yyin;           /* archivo de entrada del scanner            */

    void yyerror(const char *s); /* manejador de errores de sintaxis          */

    /* Provista por el módulo del scanner (lexicalAnalyzer.l): imprime las tres
     * tablas de símbolos (identificadores, números, strings). */
    void print_symbol_tables(void);

    /* Contador global de errores de sintaxis (lo incrementa yyerror).         */
    static int syntax_error_count = 0;

    /* ----------------------------------------------------------------------
     *  Actualización semántica recogida DURANTE el análisis sintáctico.
     *  El parser, a diferencia del scanner, conoce el contexto gramatical, por
     *  lo que puede clasificar identificadores por su rol. Aquí registramos
     *  cada función definida (su nombre y su número de parámetros) al reducir
     *  la regla `funcdef`. El número de parámetros se calcula con atributos
     *  sintetizados en `param_list` ($$ = $1 + 1).
     * ---------------------------------------------------------------------- */
    #define MAX_FUNCS 256
    #define MAX_NAME  256

    typedef struct {
        char name[MAX_NAME];
        int  nparams;
    } FuncEntry;

    static FuncEntry functions[MAX_FUNCS];
    static int       func_count = 0;

    /* Registra una función reconocida sintácticamente. */
    static void register_function(const char *name, int nparams) {
        if (func_count >= MAX_FUNCS) return;          /* tope defensivo */
        strncpy(functions[func_count].name, name ? name : "?", MAX_NAME - 1);
        functions[func_count].name[MAX_NAME - 1] = '\0';
        functions[func_count].nparams = nparams;
        func_count++;
    }

    /* Imprime el resumen de funciones reconocidas (salida del parser). */
    static void print_functions(void) {
        int i;
        printf("Funciones reconocidas (%d):\n", func_count);
        for (i = 0; i < func_count; i++) {
            printf("  - %-20s (%d parametro%s)\n",
                   functions[i].name,
                   functions[i].nparams,
                   functions[i].nparams == 1 ? "" : "s");
        }
    }
%}

/* ==========================================================================
 *  Valores semánticos asociados a tokens y no terminales.
 *    entry  : conteos / números de entrada en tablas de símbolos.
 *    lexeme : texto del lexema (lo usa IDENTIFIER para el nombre de función).
 * ========================================================================== */
%union {
    int   entry;
    char *lexeme;
}

/* --- Tokens con valor semántico --- */
%token <lexeme> IDENTIFIER

/* --- Palabras reservadas (sin valor) --- */
%token DEF FOR IF ELSE ELIF WHILE IMPORT AS IN
%token TRUE FALSE NONE RETURN BREAK CONTINUE PASS
%token AND OR NOT IS ASSERT FROM TRY EXCEPT GLOBAL RAISE DEL

/* --- Literales --- */
%token SCIENTIFIC FLOAT INT STRING

/* --- Operadores aritméticos --- */
%token PLUS MINUS MULTIPLY DIVIDE POWER FLOOR_DIV MODULO

/* --- Comparación y asignación --- */
%token EQUAL LESS_THAN GREATER_THAN NOT_EQUAL
%token LESS_EQUAL GREATER_EQUAL DOUBLE_EQUAL

/* --- Bitwise --- */
%token BIT_AND BIT_OR BIT_NOT BIT_XOR LEFT_SHIFT RIGHT_SHIFT

/* --- Delimitadores --- */
%token DOT COMMA COLON AT_SIGN
%token OPEN_PAREN CLOSE_PAREN OPEN_BRACKET CLOSE_BRACKET OPEN_BRACE CLOSE_BRACE

/* --- Tokens estructurales NUEVOS (emitidos por el scanner modificado) --- */
%token NEWLINE INDENT DEDENT

/* --- No terminales con valor (conteo de parámetros) --- */
%type <entry> param_list param_list_opt

%start program

%%

/* ==========================================================================
 *  ESTRUCTURA DEL PROGRAMA
 *  El scanner suprime el NEWLINE de líneas en blanco y de comentarios, por lo
 *  que entre sentencias no llegan NEWLINE sueltos: la lista es limpia.
 * ========================================================================== */
program
    : stmt_list
    ;

stmt_list
    : stmt_list statement
    | statement
    ;

statement
    : simple_stmt
    | compound_stmt
    | error NEWLINE   { yyerrok; }   /* recuperación: re-sincroniza al fin de línea */
    ;

/* ==========================================================================
 *  SENTENCIAS SIMPLES  (cada una termina en NEWLINE)
 * ========================================================================== */
simple_stmt
    : small_stmt NEWLINE
    ;

small_stmt
    : expr_stmt
    | return_stmt
    | pass_stmt
    | break_stmt
    | continue_stmt
    | import_stmt
    | global_stmt
    | del_stmt
    | assert_stmt
    | raise_stmt
    ;

/* Sentencia de expresión y asignación.
 * El lado izquierdo se reconoce como `testlist` general: esto admite el
 * desempaque de tuplas (a, b = ...) y la asignación encadenada (a = b = c).
 * Validar que el destino sea asignable es tarea de la fase semántica. */
expr_stmt
    : testlist
    | testlist EQUAL assign_tail
    ;

assign_tail
    : testlist
    | testlist EQUAL assign_tail
    ;

return_stmt
    : RETURN
    | RETURN testlist
    ;

pass_stmt     : PASS     ;
break_stmt    : BREAK    ;
continue_stmt : CONTINUE ;

global_stmt   : GLOBAL name_list ;
del_stmt      : DEL testlist ;

assert_stmt
    : ASSERT test
    | ASSERT test COMMA test
    ;

raise_stmt
    : RAISE
    | RAISE test
    ;

name_list
    : IDENTIFIER
    | name_list COMMA IDENTIFIER
    ;

/* ==========================================================================
 *  SENTENCIAS COMPUESTAS Y BLOQUES
 *  `suite` en una línea solo admite `simple_stmt` (no compuestas): por eso un
 *  `if` anidado obliga a un bloque INDENT/DEDENT delimitado, y NO existe
 *  ambigüedad de dangling-else.
 * ========================================================================== */
compound_stmt
    : if_stmt
    | for_stmt
    | while_stmt
    | funcdef
    | decorated
    ;

suite
    : simple_stmt
    | NEWLINE INDENT stmt_list DEDENT
    ;

if_stmt
    : IF test COLON suite elif_clauses else_clause
    ;

elif_clauses
    : /* vacío (epsilon) */
    | elif_clauses ELIF test COLON suite
    ;

else_clause
    : /* vacío (epsilon) */
    | ELSE COLON suite
    ;

/* El objetivo del for es `exprlist` (restringido, sin comparaciones) para que
 * el token IN de `for i in xs` no choque con el IN de la comparación `a in b`. */
for_stmt
    : FOR exprlist IN testlist COLON suite else_clause
    ;

while_stmt
    : WHILE test COLON suite else_clause
    ;

/* ==========================================================================
 *  DEFINICIÓN DE FUNCIONES Y DECORADORES
 *  Acción semántica: al reducir la función completa, registramos su nombre
 *  ($2) y su número de parámetros ($4, sintetizado en param_list).
 * ========================================================================== */
funcdef
    : DEF IDENTIFIER OPEN_PAREN param_list_opt CLOSE_PAREN COLON suite
        { register_function($2, $4); }
    ;

decorated
    : decorators funcdef
    ;

decorators
    : decorator
    | decorators decorator
    ;

decorator
    : AT_SIGN dotted_name NEWLINE
    | AT_SIGN dotted_name OPEN_PAREN arg_list_opt CLOSE_PAREN NEWLINE
    ;

param_list_opt
    : /* vacío (epsilon) */   { $$ = 0;  }
    | param_list              { $$ = $1; }
    ;

param_list
    : param                   { $$ = 1;      }
    | param_list COMMA param  { $$ = $1 + 1; }
    ;

param
    : IDENTIFIER                          /* a                        */
    | IDENTIFIER COLON test               /* BLOCK_SIZE: tl.constexpr */
    | IDENTIFIER EQUAL test               /* a = 0                    */
    | IDENTIFIER COLON test EQUAL test    /* a: int = 0               */
    | MULTIPLY IDENTIFIER                 /* *args   (* = MULTIPLY)   */
    | POWER IDENTIFIER                    /* **kwargs (** = POWER)    */
    ;

dotted_name
    : IDENTIFIER
    | dotted_name DOT IDENTIFIER
    ;

/* ==========================================================================
 *  IMPORTS
 * ========================================================================== */
import_stmt
    : IMPORT dotted_as_names
    | FROM dotted_name IMPORT import_targets
    ;

dotted_as_names
    : dotted_as_name
    | dotted_as_names COMMA dotted_as_name
    ;

dotted_as_name
    : dotted_name
    | dotted_name AS IDENTIFIER
    ;

import_targets
    : MULTIPLY                                   /* from x import *      */
    | import_as_names
    | OPEN_PAREN import_as_names CLOSE_PAREN
    ;

import_as_names
    : import_as_name
    | import_as_names COMMA import_as_name
    ;

import_as_name
    : IDENTIFIER
    | IDENTIFIER AS IDENTIFIER
    ;

/* ==========================================================================
 *  EXPRESIONES  -  CASCADA DE PRECEDENCIA (inambigua por construcción).
 *  De MENOR a MAYOR precedencia. Cada nivel es recursivo por la izquierda
 *  (asociatividad izquierda), salvo `power` (derecha) y `factor` (unario).
 *  Las "producciones unitarias" (or_test -> and_test -> ...) se conservan a
 *  propósito: codifican la precedencia. Eliminarlas reintroduciría ambigüedad.
 * ========================================================================== */
test
    : or_test
    ;

or_test
    : and_test
    | or_test OR and_test
    ;

and_test
    : not_test
    | and_test AND not_test
    ;

not_test
    : NOT not_test
    | comparison
    ;

comparison
    : bitor_expr
    | comparison comp_op bitor_expr
    ;

comp_op
    : LESS_THAN
    | GREATER_THAN
    | DOUBLE_EQUAL
    | NOT_EQUAL
    | LESS_EQUAL
    | GREATER_EQUAL
    | IN
    | NOT IN        /* `not in` */
    | IS
    | IS NOT        /* `is not` (NOT y ~ son tokens distintos, así que no choca) */
    ;

bitor_expr
    : bitxor_expr
    | bitor_expr BIT_OR bitxor_expr
    ;

bitxor_expr
    : bitand_expr
    | bitxor_expr BIT_XOR bitand_expr
    ;

bitand_expr
    : shift_expr
    | bitand_expr BIT_AND shift_expr
    ;

shift_expr
    : arith_expr
    | shift_expr LEFT_SHIFT arith_expr
    | shift_expr RIGHT_SHIFT arith_expr
    ;

arith_expr
    : term
    | arith_expr PLUS term
    | arith_expr MINUS term
    ;

term
    : factor
    | term MULTIPLY factor
    | term DIVIDE factor
    | term MODULO factor
    | term FLOOR_DIV factor
    ;

factor
    : PLUS factor       /* +x  unario */
    | MINUS factor      /* -x  unario */
    | BIT_NOT factor    /* ~x  unario */
    | power
    ;

power
    : atom_expr
    | atom_expr POWER factor   /* ** asociativo por la derecha */
    ;

/* Postfijos: llamadas, indexado y acceso a atributos.
 * Reconoce el launch de kernel  name[grid](args)  como
 * atom -> subíndice [ ] -> llamada ( ). */
atom_expr
    : atom
    | atom_expr OPEN_PAREN arg_list_opt CLOSE_PAREN
    | atom_expr OPEN_BRACKET subscript_list CLOSE_BRACKET
    | atom_expr DOT IDENTIFIER
    ;

/* ---- Argumentos de llamada (posicionales y por palabra clave) ---- */
arg_list_opt
    : /* vacío (epsilon) */
    | arg_list
    ;

arg_list
    : argument
    | arg_list COMMA argument
    ;

argument
    : test                       /* posicional                  */
    | IDENTIFIER EQUAL test       /* por palabra clave: mask=mask */
    | MULTIPLY test               /* *args                       */
    | POWER test                  /* **kwargs                    */
    ;

/* ---- Subíndices (incluye slice básico a:b) ---- */
subscript_list
    : subscript_item
    | subscript_list COMMA subscript_item
    ;

subscript_item
    : test
    | test_opt COLON test_opt     /* slice: a:b, :n, a: */
    ;

test_opt
    : /* vacío (epsilon) */
    | test
    ;

/* ==========================================================================
 *  ÁTOMOS Y AGREGADOS
 * ========================================================================== */
atom
    : IDENTIFIER
    | INT
    | FLOAT
    | SCIENTIFIC
    | STRING
    | TRUE
    | FALSE
    | NONE
    | OPEN_PAREN CLOSE_PAREN                 /* () tupla vacía          */
    | OPEN_PAREN testlist CLOSE_PAREN        /* (a) grupo / (a,b) tupla */
    | OPEN_BRACKET CLOSE_BRACKET             /* [] lista vacía          */
    | OPEN_BRACKET testlist CLOSE_BRACKET    /* [a, b] lista            */
    | OPEN_BRACE CLOSE_BRACE                 /* {} dict vacío           */
    | OPEN_BRACE dict_items CLOSE_BRACE      /* {k: v, ...} dict        */
    ;

/* Lista restringida para objetivos de asignación/for: hasta bitor_expr,
 * sin nivel de comparación (evita el conflicto del token IN). */
exprlist
    : bitor_expr
    | exprlist COMMA bitor_expr
    | exprlist COMMA              /* coma final */
    ;

/* Lista general de expresiones (RHS, índices, contenido de tuplas/listas). */
testlist
    : test
    | testlist COMMA test
    | testlist COMMA             /* coma final: (n_rows,) */
    ;

dict_items
    : dict_item
    | dict_items COMMA dict_item
    | dict_items COMMA
    ;

dict_item
    : test COLON test
    ;

%%

/* ==========================================================================
 *  CÓDIGO DE USUARIO
 * ========================================================================== */

/* Manejador de errores de sintaxis. Bison lo llama automáticamente al detectar
 * una construcción inválida. Reporta a stderr con el número de línea para no
 * contaminar la salida estándar del parser. */
void yyerror(const char *s) {
    syntax_error_count++;
    fprintf(stderr, "Error de sintaxis: %s (linea %d)\n", s, yylineno);
}

/* Punto de entrada. Abre el archivo de entrada (si se da), corre el parser y
 * reporta el resultado. La recuperación de errores `error NEWLINE` permite que
 * el parser continúe y reporte varios errores en una sola ejecución. */
int main(int argc, char *argv[]) {
    int result;

    if (argc >= 2) {
        yyin = fopen(argv[1], "r");
        if (yyin == NULL) {
            fprintf(stderr, "Error: no se pudo abrir el archivo de entrada: %s\n", argv[1]);
            return 1;
        }
    }   /* si no hay argumento, lee de stdin */

    result = yyparse();

    if (syntax_error_count == 0) {
        printf("Analisis sintactico exitoso: la entrada es un kernel Triton valido.\n");
        print_functions();       /* actualizacion semantica recogida durante el parseo */
        print_symbol_tables();   /* tablas de simbolos pobladas durante el escaneo      */
    } else {
        printf("Analisis sintactico finalizado con %d error(es) de sintaxis.\n",
               syntax_error_count);
    }

    if (argc >= 2 && yyin) fclose(yyin);
    return (syntax_error_count == 0) ? 0 : 1;
}
