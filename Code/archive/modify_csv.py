# import csv 
import csv 

# open input CSV file as source 
# open output CSV file as result 
with open("imputed_ppca_group.csv", "r") as source: 
	reader = csv.reader(source) 
	
	with open("imputed_ppca_group_modified.csv", "w") as result: 
		writer = csv.writer(result) 
		for r in reader: 
			
			# Use CSV Index to remove a column from CSV 
			#r[3] = r['year'] 
			writer.writerow((r[0], r[1], r[2], r[3], r[4], r[5], r[8], r[11], r[14], r[17], r[20], r[23], r[26]))
