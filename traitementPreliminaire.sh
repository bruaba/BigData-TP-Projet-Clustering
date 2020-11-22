#!/bin/bash
#awk -F "," '{$3="";$4="";$5="";$6="";$7=""; print $0}' flag.data > flag.txt
#awk -F "," 'BEGIN{OFS=","} {$3="";$4="";$5="";$6="";$7=""; print $0}' flag.data > flag.txt
awk 'BEGIN{
		FS=OFS=","
	} 
	{
		n = $1
		n = n "," $2

		for (i = 8; i <= (NF); i++) {
			n = n ","$i
		}

		print n

	}' flag.data > flag.csv