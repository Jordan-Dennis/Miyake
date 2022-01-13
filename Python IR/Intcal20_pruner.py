from operator import add

LINE_DELIMS = [",", ",", "\n"]  # The delimeters for the columns of the .csv

with open("raw_intcal20.txt", "r") as raw_intcal20, \
    open("Intcal20.csv", "w") as pruned_intcal20:   # Opening the files

    pruned_intcal20.write("cal,d14c,d14csig\n")  # Writing the col titles

    for line in raw_intcal20:   # Looping through the file maintaining lazy iterator status
        entries = line.split(" ")   # Breaking at the current delim 
        entries = entries[2:-1]     # Slicing the columns of interest 

        if entries[0] == "cal":
            continue    # If the title col return to the top of the loop

        elif float(entries[0]) < 1750:  # Getting only the years of interest 
            entries = map(add, entries, LINE_DELIMS)    # Adding correct delims

            for entry in entries:   # Writing to the file 
                pruned_intcal20.write(entry)