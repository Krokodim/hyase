# coding = 'utf-16'

import pandas
import re

df = pandas.read_csv("data/titanic.csv", index_col = "PassengerId")
rowcount = len(df)

print "Q1: How many men and women were on the ship?" \
      " Use two numbers splitted by a space."
v1 = df.Sex.value_counts()
print "A1: {0} {1}".format(v1.male, v1.female)

print "Q2: What part of passengers survived? " \
      "Express in 0..100 percents without a percent sign, " \
      "rounded to the second decimal digit."
v2 = 100.0 * df.Survived.sum()/rowcount
print "A2: {:.2f}".format(v2)

print "Q3: What is the share of 1st class passengers among all passenger." \
      " Use the same answer format as for Q2."
v3 = 100.0 * df.Sex.value_counts()[1]/rowcount
print "A3: {:.2f}".format(v3)

print "Q4: What was the passengers' age? Calculate mean and median weights."
v41, v42 = df.Age.mean(), df.Age.median()
print "A4: {:.2f} {:.2f}".format(v41,v42)

print "Q5: Do SibSp and Parch correlate? Calculate the Pearson coef."
v5 = df.SibSp.corr(df.Parch, "pearson")
print "A5: {:.2f}".format(v5)

print "Q6: What is the most popular female first name?"


def get_first_name(row):

    first_name = row.Name.split('.')[1].strip()

    rr = re.search(r'\(([A-Z a-z]+)\)', first_name)
    if rr:
        first_name = rr.group(1)

    return first_name.split(' ')[0]


v6 = df[df.Sex == 'female'].apply(get_first_name, axis=1).mode()[0]

print "A6: {0}".format(v6)
