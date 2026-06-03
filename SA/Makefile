# ==========================================================================
#  Makefile - Analizador Sintactico (Parser) del lenguaje Triton
#  Equipo 13 - TC3002B Gpo 501
#
#  Uso:
#    make            -> compila el ejecutable ./triton_parser
#    make run        -> compila y corre sobre example.triton
#    make clean      -> borra archivos generados
# ==========================================================================
CC      = gcc
CFLAGS  = -Wall
TARGET  = triton_parser

all: $(TARGET)

# Enlaza el parser (bison) con el scanner (flex).
$(TARGET): parser.tab.c lex.yy.c
	$(CC) $(CFLAGS) parser.tab.c lex.yy.c -o $(TARGET)

# Bison genera parser.tab.c, parser.tab.h y parser.output (revisar conflictos).
parser.tab.c parser.tab.h: parser.y
	bison -d -v parser.y

# Flex necesita parser.tab.h para conocer los codigos de token.
lex.yy.c: lexicalAnalyzer.l parser.tab.h
	flex lexicalAnalyzer.l

run: $(TARGET)
	./$(TARGET) example.triton

clean:
	rm -f $(TARGET) parser.tab.c parser.tab.h parser.output lex.yy.c

.PHONY: all run clean
