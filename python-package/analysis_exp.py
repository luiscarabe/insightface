import os
import shutil
import numpy

lista_ficheros = ["exp1Arc/exp1Arc.txt"]

for fichero in lista_ficheros:
	with open(fichero, "r") as f:
		print(fichero)
		lineas = f.read().splitlines()
		tabla = numpy.zeros((3,101),dtype="int")

		sujetoOriginal = ""
		num_morph = -1
		out = [False,False,False]
		seen = False

		for (i,linea) in enumerate(lineas):
			# print(i,linea)
			if linea[0:31]=="../../../experimento1/Test-data":
				sujetoOriginal = linea.split("/")[-1].split("---")[0]
				print(linea)
				target = linea.split("/")[-1].split("---")[1]
				out = [False,False,False]

			elif linea[0:5]=="Foto:":
				num_morph = int(linea.split("\t")[-1].split(".")[0])
				if out == [True, True, True]:
					seen = True
				else:
					seen = False
				flag = False
				flag2 = False
				
			elif seen==False and (linea[0:4]=="   1" or linea[0:4]=="   2" or linea[0:4]=="   3" or linea[0:4]=="   4" or linea[0:4]=="   5"):
				lineaaux = linea.split("\t")[1]
				print(num_morph)

				if  linea[0:4]=="   1":
					if lineaaux.split(" ")[0]+"_"+lineaaux.split(" ")[1] == sujetoOriginal:
						if out == [True,False,False]:
							print(sujetoOriginal,"entra en top 3")
							tabla[1][num_morph] += 4
							tabla[2][num_morph] += 4
						elif out == [True, True, False]:
							print(sujetoOriginal,"entra en top 5")	
							tabla[2][num_morph] += 4
						elif out == [False,False,False]:
							print(sujetoOriginal,"entra en top 1")
							tabla[0][num_morph] += 4
							tabla[1][num_morph] += 4
							tabla[2][num_morph] += 4

						seen = True
					else:
						# print("Se fue1",sujetoOriginal, num_morph)
						out[0] = True

				elif (linea[0:4]=="   2" or linea[0:4]=="   3"):
					if lineaaux.split(" ")[0]+"_"+lineaaux.split(" ")[1] == sujetoOriginal:
						if out == [True, False, False]:
							print(sujetoOriginal,"entra en top 3")
							tabla[1][num_morph] += 4
							tabla[2][num_morph] += 4
						elif out == [True, True, False]:
							print(sujetoOriginal,"entra en top 5")	
							tabla[2][num_morph] += 4
						seen = True
					elif flag2 == False:
						flag2 = True
					else:
						out[1] = True
				elif (linea[0:4]=="   4" or linea[0:4]=="   5"):
					if lineaaux.split(" ")[0]+"_"+lineaaux.split(" ")[1] == sujetoOriginal:
						if out == [True,True,False]:
							print(sujetoOriginal,"entra en top 5")
							tabla[2][num_morph] += 4

						seen = True
					elif flag == False:
						flag = True
					else:
						out[2] = True


				if lineaaux.split(" ")[0]+"_"+lineaaux.split(" ")[1] == sujetoOriginal:
					seen == True
					if linea[0:4]=="   1":
						tabla[0][num_morph]

		# print (tabla)
	numpy.savetxt(fichero.split(".")[0]+"Tabla.txt", tabla, delimiter="\t",fmt='%d')