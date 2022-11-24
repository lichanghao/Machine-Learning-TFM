#coding=utf-8
#------------------------
# cellGen.py
# Author:               Changhao Li (czl478@psu.edu)
# What is it:           Create random fully connected smoothed shape as hypothetical cell boundary.
# Usage:                cellGen.py [-i filename] [-o filename]
#------------------------
import matplotlib
matplotlib.use('TkAgg') # to resolve bug on mac: need to install tk

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from skimage import color as skolor # see the docs at scikit-image.org/
from skimage import measure
from scipy.ndimage import gaussian_filter


def main():
    # input parameters
    parser = argparse.ArgumentParser(description='input parameter')
    parser.add_argument('-i', dest='infile', help='the path of input file')
    parser.add_argument('-o', dest='ofile', help='the path of output file')
    args = parser.parse_args()

    n, r, N, sigma, boundary = parameters(args.infile)

    # generate smoothed Bezier Curve
    fig = plt.figure()
    path, verts = BezierCurve(N, r, fig)
    smooth_contour = smoothing(sigma, fig)

    # fit the curve into the given boundary
    smooth_contour = scaling(smooth_contour, boundary)

    # compare smoothed and original shape
    plot(fig, path, verts, smooth_contour)

    # output
    output(args.ofile, smooth_contour)


def plot(fig, path, verts, smooth_contour):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(1,2,1)
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax1.add_patch(patch)
    ax1.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax1.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax1.axis('off') # removes the axis to leave only the shape
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(smooth_contour[:, 1], smooth_contour[:, 0], linewidth=2, c='k')
    ax2.axis('off')

    plt.show()
    return


def output(ofile, curve):
    ofs = open(ofile, "w")
    for i in range(curve.shape[0]):
        ofs.write("%e \t %e \n"%(curve[i, 0], curve[i, 1]))
    return


def search_word(FILE, sword, FIMP=0):
    ifs = open(FILE, "r")
    nline = sum(1 for line in open(FILE))
    for i in range(nline):
        line = ifs.readline()
        data = line.split()
        if len(data) == 0:
            continue
        if line[0] == "#":
            continue
        if data[0] == sword:
            if data[0] == "boundary":
                ndat = 4
                Norb = np.zeros(ndat, dtype=int)
                for i in range(ndat):
                    Norb[i] = int(data[1+i])
                return Norb
            else:
                ifs.close()
                return data[1]
    if FIMP == 1:
        print("Error: Cannot find %s in file %s"%(sword, FILE))
        sys.exit()
    return 0


def parameters(infile):
    # Default parameters
    n = 5 # Number of possibly sharp edges
    r = .4 # magnitude of the perturbation from the unit circle
    # should be between 0 and 1
    N = n*3+1 # number of points in the Path
    sigma = 7 # smoothing parameter
    boundary = [0, 1, 0, 1]
    # There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve
    if infile != None:
        n = int(search_word(infile, "n"))
        r = float(search_word(infile, "r"))
        N = n*3+1
        sigma = float(search_word(infile, "sigma"))
        boundary = search_word(infile, "boundary")
    return n, r, N, sigma, boundary


def scaling(curve, boundary):
    '''
    curve: np.ndarray with shape of (N, 2)
    boundary: np.array - [xlo, xhi, ylo, yhi]
    return: np.ndarray - scaled curve fitted in boundary
    '''
    lx, ly = findBoxLenWid(boundary)
    curve_boundary = findBoundary(curve)
    clx, cly = findBoxLenWid(curve_boundary)

    curve = curve * (lx/clx, ly/cly) * 0.9

    center = findBoxCenter(boundary)
    curve_boundary = findBoundary(curve)
    curve_boundary_center = findBoxCenter(curve_boundary)
    curve = curve + (center - curve_boundary_center)

    return curve


def findBoundary(curve):
    '''
    curve: np.ndarray with shape of (N, 2)
    return: the minimal box containing the curve
    '''
    xlo = curve.min(axis=0)[0]
    xhi = curve.max(axis=0)[0]
    ylo = curve.min(axis=0)[1]
    yhi = curve.max(axis=0)[1]
    return np.array([xlo, xhi, ylo, yhi])


def findBoxCenter(box):
    return np.array([(box[1]+box[0])/2, (box[3]+box[2])/2])


def findBoxLenWid(box):
    return (box[1]-box[0]), (box[3]-box[2])


def BezierCurve(N, r, fig):
    angles = np.linspace(0,2*np.pi,N)
    codes = np.full(N,Path.CURVE4)
    codes[0] = Path.MOVETO

    verts = np.stack((np.cos(angles),np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:,None]
    verts[-1,:] = verts[0,:] # Using this instad of Path.CLOSEPOLY avoids an unnecessary straight line
    path = Path(verts, codes)

    # fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax.add_patch(patch)

    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off') # removes the axis to leave only the shape

    path = Path(verts, codes)

    ax = fig.add_axes([0,0,1,1]) # create the subplot filling the whole figure
    patch = patches.PathPatch(path, facecolor='k', lw=2) # Fill the shape in black
    ax.axis('off')

    fig.canvas.draw()
    return path, verts


def smoothing(sigma, fig, t=0.5):
    ##### Smoothing ####
    # get the image as an array of values between 0 and 1
    data = data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    gray_image = skolor.rgb2gray(data)

    # filter the image
    smoothed_image = gaussian_filter(gray_image,sigma)
    smoothed_image = smoothed_image - smoothed_image.min(); # rescale the image
    smoothed_image = smoothed_image / smoothed_image.max(); # rescale the image

    # Retrive smoothed shape as 0.5 contour
    smooth_contour = measure.find_contours(smoothed_image[::-1,:], t)[0]
    # Note, the values of the contour will range from 0 to smoothed_image.shape[0]
    # and likewise for the second dimension, if desired,
    # they should be rescaled to go between 0,1 afterwards

    return smooth_contour

if __name__ == "__main__":
    main()