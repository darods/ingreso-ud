import tabula#the pd is the standard shorthand for pandas
import pandas as pd
import PyPDF2


#declare the path of your file
file_path = "resultados_admision_ing_sistemas_2017-2.pdf"#Convert your file
df_list = tabula.read_pdf(file_path, pages="all", pandas_options={'header': None}, multiple_tables=True)

#print(df_list)

# Imprime los elementos que tienen en las primeras filas valores nulos
tablas_sinNA = []
print("tablas con titulos")
for e in df_list:
    tablas_sinNA.append(e.dropna())

print("removiendo elementos vacíos")
i=0
for e in tablas_sinNA:
    #print("----------------------- nueva tabla / Cabeceras -------------------")
    if(len(e)==0):
        tablas_sinNA.pop(i)
    i+=1
print("imprimiendo cabazas")

tablas_raras = []
for e in tablas_sinNA:
    print("-----------------")
    print(len(e.iloc[0]))
    #lista_titulos = df_list[e][df_list[e][0].isna()]
    
    
    #if (len(lista_titulos)!=0):
    #    print("tabla #%d "% e)
    #    print(lista_titulos[2])
    #'''
    
#print(lista_titulos[1])