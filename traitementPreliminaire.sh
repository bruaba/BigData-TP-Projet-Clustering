#!/bin/bash
#awk -F "," '{$3="";$4="";$5="";$6="";$7=""; print $0}' flag.data > flag.txt
#awk -F "," 'BEGIN{OFS=","} {$3="";$4="";$5="";$6="";$7=""; print $0}' flag.data > flag.txt
awk 'BEGIN{
		FS=OFS=","
	} 
	{
		n = $2

		for (i = 8; i <= (NF); i++) {
			n = n ","$i
		}

		print n

	}' OriginData/flag.data > flag.csv

sed -i 's/green/1/g' flag.csv
sed -i 's/blue/2/g' flag.csv
sed -i 's/gold/3/g' flag.csv
sed -i 's/white/4/g' flag.csv
sed -i 's/orange/5/g' flag.csv
sed -i 's/red/6/g' flag.csv
sed -i 's/black/7/g' flag.csv
sed -i 's/brown/8/g' flag.csv