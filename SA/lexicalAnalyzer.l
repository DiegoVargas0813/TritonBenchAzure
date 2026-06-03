%{
    /* ==========================================================================
     *  lexicalAnalyzer.l  -  Scanner del lenguaje Triton, integrado con yacc.
     *  Equipo 13 - TC3002B Gpo 501 - Fase II: Análisis Sintáctico.
     *
     *  CAMBIOS RESPECTO A LA FASE I (analizador léxico autónomo):
     *    1. Cada regla hace `return` del código de token definido por bison en
     *       parser.tab.h (p. ej. `return IDENTIFIER;`), en lugar de acumular un
     *       JSON. El parser (yyparse) llama a yylex() y consume estos códigos.
     *    2. Se introducen tres tokens estructurales: NEWLINE, INDENT y DEDENT.
     *       - NEWLINE marca el fin de cada línea lógica NO vacía.
     *       - INDENT/DEDENT se generan con una PILA DE INDENTACIÓN que compara la
     *         sangría de cada línea con la del bloque actual.
     *    3. Líneas en blanco y líneas de solo-comentario se ignoran por completo
     *       (no producen NEWLINE ni cambian la indentación), tal como en Python.
     *    4. Se elimina el main() (ahora lo provee parser.y) y la maquinaria JSON.
     *       Se conservan las tres tablas de símbolos como gestión semántica.
     *
     *  TÉCNICA:
     *    - Estado de arranque de línea `BOL` (Beginning Of Line) mide la sangría.
     *    - INDENT/DEDENT pueden ser varios por línea: se ENCOLAN y se drenan uno
     *      por llamada mediante un wrapper yylex() que envuelve al scanner real
     *      (raw_lex, renombrado vía YY_DECL).
     * ========================================================================== */

    #define _GNU_SOURCE
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    #include "parser.tab.h"   /* macros de token y declaración de yylval */

    /* El scanner generado por flex se llamará raw_lex(); yylex() es el wrapper. */
    #define YY_DECL int raw_lex(void)
    int raw_lex(void);

    /* ------------------------------------------------------------------
     *  Tablas de símbolos (gestión semántica heredada de la Fase I)
     * ------------------------------------------------------------------ */
    #define MAX_ENTRIES    4096
    #define MAX_LEXEME_LEN  256

    typedef struct { int entry; char lexeme[MAX_LEXEME_LEN]; } IdentifierEntry;
    typedef struct { int entry; char lexeme[MAX_LEXEME_LEN]; char type[16]; } NumberEntry;
    typedef struct { int entry; char lexeme[MAX_LEXEME_LEN]; } StringEntry;

    static IdentifierEntry identifiers[MAX_ENTRIES];
    static NumberEntry     numbers[MAX_ENTRIES];
    static StringEntry     strings[MAX_ENTRIES];
    static int id_count = 0, num_count = 0, str_count = 0;

    /* Inserta un identificador evitando duplicados; devuelve su número de fila. */
    static int add_identifier(const char* lexeme) {
        int i;
        for (i = 0; i < id_count; i++)
            if (strcmp(identifiers[i].lexeme, lexeme) == 0) return identifiers[i].entry;
        if (id_count >= MAX_ENTRIES) { fprintf(stderr, "Identifier table overflow.\n"); exit(1); }
        identifiers[id_count].entry = id_count + 1;
        strncpy(identifiers[id_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        identifiers[id_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        return ++id_count;
    }

    /* Inserta un número con su tipo explícito; devuelve su número de fila. */
    static int add_number(const char* lexeme, const char* type_str) {
        int i;
        for (i = 0; i < num_count; i++)
            if (strcmp(numbers[i].lexeme, lexeme) == 0) return numbers[i].entry;
        if (num_count >= MAX_ENTRIES) { fprintf(stderr, "Number table overflow.\n"); exit(1); }
        numbers[num_count].entry = num_count + 1;
        strncpy(numbers[num_count].lexeme, lexeme, MAX_LEXEME_LEN - 1);
        numbers[num_count].lexeme[MAX_LEXEME_LEN - 1] = '\0';
        strncpy(numbers[num_count].type, type_str, 15);
        numbers[num_count].type[15] = '\0';
        return ++num_count;
    }

    /* Inserta un string quitando las comillas envolventes; devuelve su fila. */
    static int add_string(const char* lexeme) {
        int i;
        size_t len = strlen(lexeme);
        const char* start = lexeme;
        size_t copy_len = len;
        if (len >= 2 && (lexeme[0] == '"' || lexeme[0] == '\'') && lexeme[len-1] == lexeme[0]) {
            start = lexeme + 1; copy_len = len - 2;
        }
        if (copy_len >= MAX_LEXEME_LEN) copy_len = MAX_LEXEME_LEN - 1;
        for (i = 0; i < str_count; i++)
            if (strncmp(strings[i].lexeme, start, copy_len) == 0 && strings[i].lexeme[copy_len] == '\0')
                return strings[i].entry;
        if (str_count >= MAX_ENTRIES) { fprintf(stderr, "String table overflow.\n"); exit(1); }
        strings[str_count].entry = str_count + 1;
        strncpy(strings[str_count].lexeme, start, copy_len);
        strings[str_count].lexeme[copy_len] = '\0';
        return ++str_count;
    }

    /* Imprime las tres tablas de símbolos. La llama el parser al terminar. */
    void print_symbol_tables(void) {
        int i;
        printf("\n--- Tablas de simbolos ---\n");
        printf("Identificadores (%d):\n", id_count);
        for (i = 0; i < id_count; i++)
            printf("  [%d] %s\n", identifiers[i].entry, identifiers[i].lexeme);
        printf("Numeros (%d):\n", num_count);
        for (i = 0; i < num_count; i++)
            printf("  [%d] %s (%s)\n", numbers[i].entry, numbers[i].lexeme, numbers[i].type);
        printf("Strings (%d):\n", str_count);
        for (i = 0; i < str_count; i++)
            printf("  [%d] %s\n", strings[i].entry, strings[i].lexeme);
    }

    /* ------------------------------------------------------------------
     *  Pila de indentación y cola de tokens estructurales
     * ------------------------------------------------------------------ */
    #define MAX_DEPTH 128
    static int indent_stack[MAX_DEPTH] = {0};  /* indent_stack[0] = 0 (nivel base) */
    static int indent_top = 0;                 /* índice del tope de la pila       */

    #define QSIZE 256
    static int q_buf[QSIZE];
    static int q_head = 0, q_tail = 0;
    static int q_empty(void) { return q_head == q_tail; }
    static void q_push(int t) { q_buf[q_tail++] = t; }
    static int  q_pop(void)  { int t = q_buf[q_head++]; if (q_head == q_tail) q_head = q_tail = 0; return t; }

    static int line_has_tokens = 0;  /* ¿la línea actual produjo algún token real? */
    static int eof_flushed     = 0;  /* ¿ya emitimos NEWLINE/DEDENT finales?        */
    static int first_call      = 1;  /* para entrar en BOL en la primera llamada    */

    /* Ancho de sangría: cuenta espacios y tabs como 1 unidad cada uno.
     * (Los kernels Triton generados por LLM usan espacios; mezclar tabs y
     *  espacios no se soporta de forma robusta y se documenta como limitación.) */
    static int indent_width(const char* s) {
        int w = 0;
        for (; *s; s++) if (*s == ' ' || *s == '\t') w++;
        return w;
    }

    /* Compara la sangría de la nueva línea con la pila y encola INDENT/DEDENT. */
    static void handle_indentation(int col) {
        if (col > indent_stack[indent_top]) {
            if (indent_top + 1 < MAX_DEPTH) {
                indent_stack[++indent_top] = col;
                q_push(INDENT);
            }
        } else if (col < indent_stack[indent_top]) {
            while (indent_top > 0 && indent_stack[indent_top] > col) {
                indent_top--;
                q_push(DEDENT);
            }
            if (indent_stack[indent_top] != col)
                fprintf(stderr, "Advertencia: indentacion inconsistente (linea %d)\n", yylineno);
        }
        /* col == tope: mismo bloque, no se encola nada. */
    }

    /* Marca que la línea produjo un token y lo devuelve al parser. */
    #define RET(tok) do { line_has_tokens = 1; return (tok); } while (0)
%}

%option noyywrap yylineno
%x BOL

LETTER  [a-zA-Z_]
DIGIT   [0-9]
ID      {LETTER}({LETTER}|{DIGIT})*

SCI     {DIGIT}+(\.[0-9]+)?[eE][+-]?{DIGIT}+
FLOAT   {DIGIT}+\.{DIGIT}+|{DIGIT}+\.|\.{DIGIT}+
INT     {DIGIT}+

STR_DQ  \"([^\\\n\"]|\\.)*\"
STR_SQ  \'([^\\\n\']|\\.)*\'

%%

%{
    /* Se ejecuta al inicio de cada llamada a raw_lex; entra en BOL solo la 1ra vez. */
    if (first_call) { first_call = 0; BEGIN(BOL); }
%}

    /* ============== INICIO DE LÍNEA (medición de sangría) ============== */
<BOL>[ \t]*\r?\n            { /* línea en blanco: se ignora por completo */ }
<BOL>[ \t]*#[^\n]*\r?\n     { /* línea de solo-comentario: se ignora     */ }
<BOL>[ \t]*#[^\n]*          { /* comentario al final del archivo sin \n  */ }
<BOL>[ \t]*[^ \t\r\n#]      { /* línea real: medir sangría (incluye el 1er carácter real,
                                que se devuelve al flujo con yyless para re-escanearlo).
                                Esto evita un match vacío, que flex no dispararía. */
                               int col = indent_width(yytext);
                               yyless(yyleng - 1);     /* devuelve el primer carácter real */
                               BEGIN(INITIAL);
                               handle_indentation(col);
                               if (!q_empty()) return q_pop();
                             }
<BOL>[ \t]+                 { /* solo espacios al final del archivo: se ignora */ }

    /* ============== CUERPO DE LÍNEA (tokens normales) ============== */
[ \t\r]+    ;                       /* espacios entre tokens: se ignoran */
#[^\n]*     ;                       /* comentario tras código en la línea */

\r?\n       { /* fin de línea lógica */
              if (line_has_tokens) {
                  line_has_tokens = 0;
                  BEGIN(BOL);
                  return NEWLINE;
              }
              BEGIN(BOL);            /* la línea no tuvo tokens: no emitir NEWLINE */
            }

"def"       { RET(DEF);      }
"for"       { RET(FOR);      }
"if"        { RET(IF);       }
"else"      { RET(ELSE);     }
"elif"      { RET(ELIF);     }
"while"     { RET(WHILE);    }
"import"    { RET(IMPORT);   }
"as"        { RET(AS);       }
"in"        { RET(IN);       }
"True"      { RET(TRUE);     }
"False"     { RET(FALSE);    }
"None"      { RET(NONE);     }
"return"    { RET(RETURN);   }
"break"     { RET(BREAK);    }
"continue"  { RET(CONTINUE); }
"pass"      { RET(PASS);     }
"and"       { RET(AND);      }
"or"        { RET(OR);       }
"not"       { RET(NOT);      }
"is"        { RET(IS);       }
"assert"    { RET(ASSERT);   }
"from"      { RET(FROM);     }
"try"       { RET(TRY);      }
"except"    { RET(EXCEPT);   }
"global"    { RET(GLOBAL);   }
"raise"     { RET(RAISE);    }
"del"       { RET(DEL);      }

    /* Operadores compuestos antes que sus prefijos (maximal munch). */
"**"        { RET(POWER);         }
"//"        { RET(FLOOR_DIV);     }
"<<"        { RET(LEFT_SHIFT);    }
">>"        { RET(RIGHT_SHIFT);   }
"=="        { RET(DOUBLE_EQUAL);  }
"!="        { RET(NOT_EQUAL);     }
"<="        { RET(LESS_EQUAL);    }
">="        { RET(GREATER_EQUAL); }

"+"         { RET(PLUS);          }
"-"         { RET(MINUS);         }
"*"         { RET(MULTIPLY);      }
"/"         { RET(DIVIDE);        }
"%"         { RET(MODULO);        }
"="         { RET(EQUAL);         }
"<"         { RET(LESS_THAN);     }
">"         { RET(GREATER_THAN);  }
"&"         { RET(BIT_AND);       }
"|"         { RET(BIT_OR);        }
"~"         { RET(BIT_NOT);       }
"^"         { RET(BIT_XOR);       }

"."         { RET(DOT);           }
","         { RET(COMMA);         }
":"         { RET(COLON);         }
"@"         { RET(AT_SIGN);       }
"("         { RET(OPEN_PAREN);    }
")"         { RET(CLOSE_PAREN);   }
"["         { RET(OPEN_BRACKET);  }
"]"         { RET(CLOSE_BRACKET); }
"{"         { RET(OPEN_BRACE);    }
"}"         { RET(CLOSE_BRACE);   }

    /* Literales: strings, luego SCI, FLOAT, INT (maximal munch); ID al final. */
{STR_DQ}    { add_string(yytext);               RET(STRING);     }
{STR_SQ}    { add_string(yytext);               RET(STRING);     }
{SCI}       { add_number(yytext, "scientific"); RET(SCIENTIFIC); }
{FLOAT}     { add_number(yytext, "float");      RET(FLOAT);      }
{INT}       { add_number(yytext, "int");        RET(INT);        }
{ID}        { yylval.lexeme = strdup(yytext); add_identifier(yytext); RET(IDENTIFIER); }

.           { fprintf(stderr, "Error lexico: caracter no reconocido '%s' (linea %d)\n",
                      yytext, yylineno); }

%%

/* ==========================================================================
 *  yylex(): wrapper sobre raw_lex().
 *    1. Drena primero la cola de tokens estructurales pendientes.
 *    2. Al llegar al EOF, emite el NEWLINE final (si la última línea tuvo
 *       tokens) y todos los DEDENT que falten para cerrar los bloques abiertos.
 * ========================================================================== */
int yylex(void) {
    int t;

    if (!q_empty()) return q_pop();      /* INDENT/DEDENT/NEWLINE en cola */

    t = raw_lex();

    if (t == 0) {                        /* EOF alcanzado por el scanner */
        if (!eof_flushed) {
            eof_flushed = 1;
            if (line_has_tokens) { line_has_tokens = 0; q_push(NEWLINE); }
            while (indent_top > 0) { indent_top--; q_push(DEDENT); }
            if (!q_empty()) return q_pop();
        }
        return 0;                        /* fin real */
    }
    return t;
}
