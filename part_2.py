#!/usr/bin/env python3

"""___________________________________________________________________________________________
Part 2:
Building Document Vectors for Web Page Content Analysis
A document can be represented as a Bag of Words where a document is represented in a |V|
dimensional vector where |V| is a set of all the unique terms that occur in the entire collection
of the documents (called Dictionary) as the features with a weight score in Term Frequency(TF)
- the frequency of each term occurring in each document - and Inverted Document Frequency.
For this transformation, you have to build an Inverted Index (which is a Term Look_Up Table or
Term Dictionary with Term Frequency (TF) and Document Frequency (DF). See how to build an
Inverted Index for TF-IDF in Slides 10 – 14 in the Lecture Notes.
___________________________________________________________________________________________
"""
# Import the Lab4 class from the src folder

import src.Lab4 as LAB4

def main ():
    """Building Document Vectors for Web Page Content Analysis"""
    lab4_instance =  LAB4.Lab4()
    lab4_instance.part_2()

if __name__ == "__main__":
    main()