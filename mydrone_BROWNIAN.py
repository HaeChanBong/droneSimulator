import time
import random
import drawSample
import math
import _tkinter
import sys
import loader
import numpy
import sys
import cPickle
import os.path
import Tkinter as tk
import sys
import Tkinter as tk

from PIL import ImageFilter
import TileServer
import geoclass
from PIL import ImageFont

debug = 0  # debug 1 or 2 means using a very simplified setup
verbose=0  # print info (can be set on command line)
versionNumber = 1.0
zoomLevel = 18
loadStateFile = 'classifier.state'  # default file for classifier data
documentation = \
"""
  This program is a stub for your COMP 417 robotics assignment.
"""

# Variables.
tileScale = float(256/5) # Length of tile.

##########################################################################################
#########  Do non-stardard imports and print helpful diagnostics if necessary ############
#########  Look  for "real code" below to see where the real code goes        ############
##########################################################################################

missing = []
fix = ""

try:
    import scipy
    from scipy import signal
except ImportError:
    missing.append( " scipy" )
    fix = fix +  \
        """
        On Ubuntu linux you can try: sudo apt-get install python-numpy python-scipy python-matplotlib
        On OS X you can try:
              sudo easy_install pip
              sudo pip install  scipy
        """
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    missing.append( " matplotlib" )
    fix = fix +  \
        """
        On Ubuntu linux you can try: sudo apt-get install python-matplotlib
        On OS X you can try:
              sudo easy_install pip
              sudo pip install matplotlib
        """
try:
    import numpy as np
except ImportError:
     missing.append( " numpy 12.8921568627 12.5" )
     fix = fix + \
        """
          sudo easy_install pip
          sudo pip install numpy
        """
try:
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
except ImportError:
     missing.append( " scikit-learn " )
     fix = fix + \
        """
          sudo easy_install pip
          sudo pip install scikit-learn
        """
try:
    from PIL import Image
    from PIL import ImageTk
    from PIL import ImageDraw
except ImportError:
     missing.append( " PIL (more recently known as pillow) " )
     fix = fix + \
        """
          sudo easy_install pip
          sudo pip install pillow
        """
if missing:
     print "*"*60
     print "Cannot run due to missing required libraries of modules."
     print "Missing modules: "
     for i in missing: print "    ",i
     print "*"*60
     print fix
     sys.exit(1)

version = "Greg's drone v%.1f  $HGdate: Fri, 24 Nov 2017 09:38:37 -0500 $ $Revision: f330eb3280c9 Local rev 2 $" % versionNumber

print version
print " ".join(sys.argv)
##########################################################################################
#########     Parse command-line arguments   #############################################
##########################################################################################
while len(sys.argv)>1:
    if len(sys.argv)>1 and sys.argv[1]=="-v":
        verbose = verbose+1
        del sys.argv[1]
    elif len(sys.argv)>1 and sys.argv[1]=="-load":
        if len(sys.argv)>2 and not sys.argv[2].startswith("-"):
            loadStateFile = sys.argv[2]
            del sys.argv[2]
        else:
            loadStateFile = 'classifier.state'
        del sys.argv[1]
    elif len(sys.argv)>1 and sys.argv[1] in ["-h", "-help", "--help"]: # help
        print documentation
        sys.argv[1] = "-forceusagemesasge"
    else:
        print "Unknown argument:",sys.argv[1]
        print "Usage: python ",sys.argv[0]," [-h (help)][-v]    [-f TRAININGDATADIR] [-t TESTDIR] [-load [STATEFILE]]"
        sys.exit(1)

##########################################################################################
#########  "real code" is here, at last!                                      ############
##########################################################################################
# my position
tx, ty = 0,0 # This is the translation to use to move the drone
oldp = [tx,ty]  # Last point visited

fill = "white"
image_storage = [ ] # list of image objects to avoid memory being disposed of

def autodraw():
    """ Automatic draw. """
    draw_objects()
    tkwindow.canvas.after(100, autodraw)

pca, clf, classnames = geoclass.loadState()
land_matrix = numpy.zeros(shape=(20, 20), dtype=int)
visit_matrix = numpy.zeros(shape=(20, 20), dtype=int)

def is_not_urban_or_desert(x,y):
    #print x,y
    x = int(numpy.floor(x))
    y = int(numpy.floor(y))

    if (land_matrix[x][y] == 2 or land_matrix[x][y] == 1 or c == 1 or c == 2):
        return False
    else:
        return True

def is_in_bounds(x,y):
    if (0.0<=x<=20.0) and (0.0<=y<=20.0):
        return True
    else:
        return False

def are_visitable(x,y):
    # Check if urban or desert.
    neighbours = [(x-1.0,y),(x+1.0,y),(x,y+1.0),(x,y-1.0)]
    good_neighbours = []
    final = []
    for neighbour in neighbours:
        if (is_in_bounds(neighbour[0],neighbour[1])):
            if is_not_urban_or_desert(neighbour[0],neighbour[1]):
                good_neighbours.append(neighbour)
            else:
                pass
        else:
            pass

    return shuffle(good_neighbours)

def shuffle(neighbour_list):
    final_list = []
    random_used = []
    while len(random_used) != len(neighbour_list):
        random_index = random.uniform(0,len(neighbour_list))
        if random_index in random_used:
            pass
        else:
            random_used.append(random_index)
            final_list.append(neighbour_list[int(random_index)])
    return final_list

def move_to(curr_x, curr_y, dest_x,dest_y):
    tx = float((dest_x-curr_x)*tileScale)
    ty = float((dest_y-curr_y)*tileScale)

    return tx,ty

def pick_next_brownian(x,shapey):
    neighbours = [(x-1,y),(x+1,y),(x,y+1),(x,y-1)]
    least_visited = 10000
    least_visited_neighbour = (0,0)
    for neighbour in neighbours:
        if visit_matrix(neighbour) <= least_visited:
            least_visited = visit_matrix(neighbour)
            least_visited_neighbour = neighbour
    return least_visited_neighbour

def pick_next_sweep_bound(x,y,bounds):
    if bounds[0]:
        next_x = x+1
        next_y = y
    if bounds[1]:
        next_x = x-1
        next_y = y
    if bounds[2]:
        next_x = x
        next_y = y-1
    if bounds[3]:
        next_x = x
        next_y = y+1
    else:
        next_x = x
        next_y = y
    return next_x,next_y

def pick_next_sweep(x, y, good_neighbours, prev_pt):
    least_visited = 10000
    least_visited_neighbour = prev_pt
    for neighbour in good_neighbours:
        if ((visit_matrix[int(numpy.floor(neighbour[0])),int(numpy.floor(neighbour[1]))]) <= least_visited):
            least_visited = visit_matrix[int(numpy.floor(neighbour[0])),int(numpy.floor(neighbour[1]))]
            least_visited_neighbour = neighbour
    return least_visited_neighbour

def unvisited(x, y):
    if land_matrix[x][y] > 3:
        return True
    else:
        return False
def get_numbers(land_matrix, visit_matrix):
    arable = 0
    water = 0
    urban = 0
    unique = 0
    desert = 0
    total = 0

    for x in range(0,19):
        for y in range(0,19):
            if visit_matrix[x][y] == 1:
                unique += 1
            total += visit_matrix[x][y]
            if land_matrix[x][y] == 4:
                arable += 1
            elif land_matrix[x][y] == 1:
                desert += 1
            elif land_matrix[x][y] == 2:
                urban += 1
            elif land_matrix[x][y] == 3:
                water += 1
    things = {"arable": arable, "water": water, "urban": urban, "unique": unique, "total": total}
    return things

def draw_objects():
    """ Draw target balls or stuff on the screen. """
    global tx, ty, maxdx, maxdy, unmoved, tileScale, directionH, directionV, txt, c, temp_x, temp_y
    global oldp, prev_pt, urban
    global objectId
    global ts # tileSerscalexver
    global actual_pX, actual_pY
    global fill
    global scalex, scaley  # scale factor between out picture and the tileServer

    mat_x = float(oldp[0])/float(256/scalex)
    mat_y = float(oldp[1])/float(256/scaley)

    #tkwindow.canvas.move( objectId, int(tx-MYRADIUS)-oldp[0],int(ty-MYRADIUS)-oldp[1] )
    if unmoved:
        # initialize on first time we get here
        unmoved=0
        tx,ty = tileScale, 0.0
    else:
        visitable_neighbours = are_visitable(mat_x, mat_y)

        # Have good nieghbors.
        if len(visitable_neighbours) > 1:
            next_move = pick_next_sweep(mat_x,mat_y,visitable_neighbours, prev_pt)
            visit_matrix[int(numpy.floor(next_move[0])),int(numpy.floor(next_move[1]))] += 1
            tx,ty = move_to(mat_x,mat_y,next_move[0], next_move[1])
        # Don't have good nieghbors.
        else:
            visit_matrix[int(prev_pt[0]), int(prev_pt[1])] += 1
            visit_matrix[int(mat_x), int(mat_y)] += 1
            land_matrix[int(mat_x), int(mat_y)] = c
            tx,ty = move_to(mat_x, mat_y, prev_pt[0], prev_pt[1])

        # Save previous pt.
        prev_pt = [mat_x, mat_y]

    # draw the line showing the path
    tkwindow.polyline([oldp, [oldp[0]+tx, oldp[1]+ty]], style=5, tags=["path"]  )
    tkwindow.canvas.move( objectId, tx,ty )

    # update the drone position
    oldp = [oldp[0]+tx, oldp[1]+ty]

    # map drone location back to lat, lon
    # This transforms pixels to WSG84 mapping, to lat,lon
    lat,lon = ts.imagePixelsToLL( actual_pX, actual_pY, zoomLevel,  oldp[0]/(256/scalex), oldp[1]/(256/scaley) )

    # get the image tile for our position, using the lat long we just recovered
    im, foox, fooy, fname = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, 1, 1, 0, 0)

    # Use the classifier here on the image "im"
    im2 = np.asarray(im,dtype=np.float32)
    im2 = np.ravel(im)
    c,txt = geoclass.classifyOne(pca,clf,im2,target_names = classnames)
    x = int(numpy.floor(mat_x))
    y = int(numpy.floor(mat_y))
    if c == 0:
        land_matrix[x][y] = 4
    else:
        land_matrix[x][y] = c

    # This is the drone, let's move it around
    tkwindow.canvas.itemconfig(objectId, tag='userball', fill=fill)
    tkwindow.canvas.drawn = objectId

    #  Take the tile and shrink it to go in the right place
    im = im.resize((int(im.size[0]/scalex),int(im.size[1]/scaley)))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("Raleway-ThinItalic.ttf",30)
    draw.text( (0,0),txt[0],(0,255,0),font=font)
    im.save("/tmp/locationtile.gif")
    photo = tk.PhotoImage(file="/tmp/locationtile.gif" )
    tkwindow.image = tkwindow.canvas.create_image( 256/scalex*int(oldp[0]/(256/scalex)), 256/scalex*int(oldp[1]/(256/scalex)), anchor=tk.NW, image=photo, tags=["tile"] )
    image_storage.append( photo ) # need to save to avoid garbage collection

    # This arrenges the stuff being shown
    tkwindow.canvas.lift( objectId )
    tkwindow.canvas.tag_lower( "tile" )
    tkwindow.canvas.tag_lower( "background" )
    tkwindow.canvas.pack()

    # Code to move the drone can go here
    # Move a small amount by changing tx,ty
    tx = 1.0
    ty = 1.0

# MAIN CODE. NO REAL NEED TO CHANGE THIS

ts = TileServer.TileServer()

# Top-left corner of region we can see

lat, lon = 45.44203, -73.602995    # verdun

# Size of region we can see, measure in 256-goepixel tiles.  Geopixel tiles are what
# Google maps, bing, etc use to represent the earth.  They make up the atlas.
#
tilesX = 20
tilesY = 20
tilesOffsetX = 0
tilesOffsetY = 00

# Get tiles to cover the whole map (do not really need these at this point, be we cache everything
# at the biginning this way, and can draw it all.
# using 1,1 instead of tilesX, tilesY to see just the top left image as a check
#
#actual, actual_pX, actual_pY, fname = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, 1, 1, tilesOffsetX, tilesOffsetY)
actual, actual_pX, actual_pY, fname = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, tilesX, tilesY, tilesOffsetX, tilesOffsetY)

# Rather than draw the real data, we can use a white map to see what is unexplored.
bigpic = Image.new("RGB", (256*tilesX, 256*tilesY), "white")
bigpic.paste(actual, (0,0))  # paste the actual map over the pic.

# How to draw a rectangle.
# You should delete or comment out the next 3 lines.
draw = ImageDraw.Draw(bigpic)
xt,yt = 0,0
draw.rectangle(((xt*256-1, yt*256-1), (xt*256+256+1, yt*256+256+1)), fill="red")

# put in image

# Size of our on-screen drawing is arbitrarily small
myImageSize = 1024
scalex = bigpic.size[0]/myImageSize  # scale factor between our picture and the tileServer
scaley = bigpic.size[1]/myImageSize  # scale factor between our picture and the tileServer
im = bigpic.resize((myImageSize,myImageSize))
im = im.filter(ImageFilter.BLUR)
im = im.filter(ImageFilter.BLUR)

im.save("mytemp.gif") # save the image as a GIF and re-load it does to fragile nature of Tk.PhotoImage
tkwindow  = drawSample.SelectRect(xmin=0,ymin=0,xmax=1024 ,ymax=1024, nrects=0, keepcontrol=0 )#, rescale=800/1800.)
root = tkwindow.root

root.title("Drone simulation")

# Full background image
photo = tk.PhotoImage(file="mytemp.gif")
tkwindow.imageid = tkwindow.canvas.create_image( 0, 0, anchor=tk.NW, image=photo, tags=["background"] )
image_storage.append( photo )
tkwindow.canvas.pack()

tkwindow.canvas.pack(side = "bottom", fill = "both",expand="yes")

MYRADIUS = 7
MARK="mark"

# Place our simulated drone on the map
sx,sy=12.5*tileScale, 12.5*tileScale # over the river
prev_pt = [sx,sy]
#sx,sy = 220,280 # over the canal in Verdun, mixed environment
oldp = [sx,sy]
objectId = tkwindow.canvas.create_oval(int(sx-MYRADIUS),int(sy-MYRADIUS), int(sx+MYRADIUS),int(sy+MYRADIUS),tag=MARK)
unmoved = 1

# initialize the classifier
# We can use it later using these global variables.
#
pca, clf, classnames = geoclass.loadState( loadStateFile, 1.0)

# launch the drawing thread
autodraw()

#Start the GUI
root.mainloop()
