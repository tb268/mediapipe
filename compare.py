import csv

def main():


    with open("posedata.csv", "w") as f1,open("posedata2.csv", "w") as f2:
        reader1=csv.reader(f1)
        reader2=csv.reader(f2)
        pass

        for (x,y) in zip(reader1,reader2):
            x=x.split()
            pass


if __name__ == '__main__':
    main()