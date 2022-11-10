#!/usr/bin/env python3

"""
Part 1:
Building a Knowledge Base in Inverted Index for Big Data Applications such as
Web Page Categorization or Google Search Engine

Build an Inverted Index in MySql tables (or any SQL Server) on the collection of the State
Union Address Texts with the following simplified scheme as seen the slide 13 of the Inverted
Lecture Note covered in class below.
Build a Full Inverted Index Structure as given below:
Doc_Id: Name of President Combined with Date of Union Address
First Level Dictionary Table (Term, DocFreq, CollectionFreq)
Second Level Posting Table (Term, Doc_Id, TermFreq) 

You are to create two index files to be created as tables in a SQL Server as shown in the
Lecture Note on Inverted Index.
https://eecs.csuohio.edu/~sschung/cis612/InvertedIndex.pdf 

"""
# Import the Lab4 class from the src folder

import src.Lab4 as LAB4

def main ():
    """Building a Knowledge Base in Inverted Index for Big Data Applications"""
    lab4_instance =  LAB4.Lab4()
    lab4_instance.part_1()

if __name__ == "__main__": 
    main()