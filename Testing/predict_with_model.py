import sys
import logging

from clarifai.rest import Image as ClImage
from clarifai.rest import ClarifaiApp

app = ClarifaiApp("q2X8H5j09-03K90Q5VyUqrJMWrRznqkDnkfOyPRI", "Pr0698K9_8tEfMV9Nhq6UPWEpXFs8QnYQSElJA_8")

model = app.models.get('Lung_Cancer')


f = open('out.txt', 'w')

# predict with samples
#print model.predict_by_filename('./positive/78813f0f98d4dd843ce735589746e3ad.jpg')
#print "\n\n\n\n\n"
#print model.predict_by_filename('./negative/930892d7c3ecd8bac2c9028841a697ed.jpg')

# positive example
print >> f, "Positive image test"
print >> f, model.predict_by_filename('./positive/78813f0f98d4dd843ce735589746e3ad.jpg')
print >> f, "\n\n\n"

# negative example
print >> f, "Negative image test"
print >> f, model.predict_by_filename('./negative/930892d7c3ecd8bac2c9028841a697ed.jpg')
f.close()