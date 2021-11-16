# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import os
os.chdir(r'C:\Users\smeyer')
from pyautocad import Autocad, APoint
import array




acad = Autocad(create_if_not_exists=True)

acad.prompt("Hello, Autocad from Python")

print(acad.doc.Name)

#pattern 1
omega = 2*np.pi
v_n = 10
end_ang = 12*np.pi
unit_ang = np.pi/100
theta = np.linspace(0, end_ang, np.round(end_ang/unit_ang).astype(int))
x = v_n*theta/omega*np.cos(theta)
y = v_n*theta/omega*np.sin(theta)
z = 0.1*np.arange(len(x))
#pattern 2
#x = np.linspace(-10, 10, num=100)
#y = np.sin(x)
#z = x**2+2*x+1

n_points = len(x)
points_2d = np.array([x,y,z]).T.flatten()
points_double = array.array("d", points_2d)
acad.model.Add3dpoly(points_double)

# for i in range(n_points):
#   p1 = APoint(x[i], y[i])
#   if i<(n_points-1):
#       p2 = APoint(x[i+1], y[i+1])
#       acad.model.Addline(p1, p2)

# p0 = APoint(10, 10)
# arc = acad.model.AddArc(p), 10, 0, np.pi/2)

dp = APoint(0, 0.5, 0)

for i in range(n_points):
    p1 = APoint(x[i], y[i], z[i])
    if(i%5) == 0:
        text = acad.model.AddText('Point %d' %i, p1+dp, 0.2)
        acad.model.AddCircle(p1, 0.1)

for obj in acad.iter_objects(['Circle', 'line']):
    print(obj.ObjectName)
    