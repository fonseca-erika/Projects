#include "Tools.h"

/* print Fahrenheit-Celsius table */


int main()
{

	/* 
	fahr_to_celsius();
	celsius_to_fahr();
	celsius_to_fahr_reverse_order();
	caractere();
	my_pointers();
	*/
	int a = 50, b = 500;
	swap(&a, &b);
	//char mes = month_name(1);

	struct student erika = { "Erika", 37, 1 };
	

	int* pt1;

	int matrix[10][5];
	pt1 = matrix;
	int valor = 10;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 5; j++) {
			matrix[i][j] = valor;
			valor = valor + 1;
		}
	}

	printf("pointer %d \n", pt1);
	printf("pointer content %d \n", *pt1);


	return 0;
}


