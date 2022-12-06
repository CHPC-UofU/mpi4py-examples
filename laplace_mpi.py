#!/usr/bin/env python3
# laplace_mpi.py - MPI implementation of Laplace problem from XSEDE summer bootcamp.
# Author: Brett Milash, brett.milash@utah.edu.
# This is a python / mpi4py solution to the Laplace exercise from the 
# XSEDE summer bootcamp: https://www.psc.edu/wp-content/uploads/2021/06/MPI-Laplace-Exercise-Intro.pdf

import numpy as np
import math
import sys
import matplotlib
import matplotlib.pyplot as plt
import argparse

from mpi4py import MPI

def ReportProgressText(a,iteration,dt,arrayname):
    "Simple text display of array."
    print(f"Array {arrayname} at iteration {iteration:3d} dt {dt:3.4f}")
    print(a)

def ReportProgressGraphical(a,iteration,dt,arrayname):
    "Heatmap display of array."
    global fig, ax
    im = ax.imshow(a)
    title=f"Array {arrayname} at iteration {iteration:3d} dt {dt:3.4f}"
    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def LaplaceSolverOverlappedArrays(comm,max_temp,max_iteration,min_temp_change,temp,verbose):
    """
    Solves laplace problem using overlapped arrays to replace nested loops.
    """
    rank = comm.Get_rank()
    size=comm.Get_size()
    #print(f"Rank {rank} of {size} processes.")
    global_dt=[max_temp]*size    # this is delta-t across all slices.
    iteration=0
    temp_next=np.copy(temp)
    (anumrows,anumcols)=temp.shape
    dim=anumcols-2    # dim is the number of data columns excluding margins.

    # Scatter the data across all the processes.
    if rank == 0:
        # Scatter data to other processes. Slice the array into
        # "size" equal slices, and load those slices into the
        # list "data".
        data=[]
        m=int(dim/size)    # m is the number of data columns per slice.
        starts=np.linspace(0,dim,size,endpoint=False,dtype=int)
        ends=starts+m+2    # the +2 includes the margins.
        for (start,end) in zip(starts,ends):
            data.append(temp[:,start:end])
    else:
        data=None
    
    # Do the scatter.
    data=comm.scatter(data,root=0)
    data_next=data.copy()

    # Testing: report on the results of the scatter and exit.
    #print(f"Rank {rank}: data[-1,]: {data[-1:]}, type(data): {type(data)}")
    #sys.exit(0)
    (anumrows,anumcols)=data.shape

    while max(global_dt) > min_temp_change and iteration < max_iteration:
        iteration+=1
        if verbose and rank==0:
            print(f"global_dt: {np.array(global_dt)}")

        # Calculate temp at next time point. Each cell's temp is average
        # of its 4 neighbors.
        north=data[0:anumrows-2,1:anumcols-1]    # North
        south=data[2:anumrows-0,1:anumcols-1]    # South
        east=data[1:anumrows-1,2:anumcols-0]    # East
        west=data[1:anumrows-1,0:anumcols-2]    # West
        m = np.mean( [ north, south, east, west, ], axis=0)
        data_next[1:anumrows-1,1:anumcols-1] = m.copy()

        # Calculate delta-t
        dt=np.amax(data_next-data)
        data=np.copy(data_next)

        # Exchange data at margin of array with other processes.
        if rank in list(range(0,size-1)):
            # Send column -2 to rank+1
            comm.isend(data[:,-2],dest=rank+1)
        if rank in list(range(1,size)):
            data[:,0]=comm.recv(source=rank-1)

        if rank in list(range(1,size)):
            # Send column 0 to rank-1
            comm.isend(data[:,1],dest=rank-1)
        if rank in list(range(0,size-1)):
            data[:,-1]=comm.recv(source=rank+1)

        if rank == size-1 and verbose and iteration % 10 == 0:
            ReportProgressText(data,iteration,dt,"data")

        # Exchange dt with all other processes so the max global
        # dt can be calculated.
        global_dt=comm.allgather(dt)

    # Gather slices back into process rank 0, excluding the margins.
    data=data[1:-1,1:-1].copy()
    slices=comm.gather(data,root=0)
    if rank==0:
        result=np.concatenate(slices,axis=1)
        dt=max(global_dt)
        ReportProgressText(result,iteration,dt,"result")
    else:
        sys.exit(0)

    return (dt,iteration)

def LaplaceSolverNestedLoops(max_temp,max_iteration,min_temp_change,temp,verbose):
    """
    Solves laplace problem using nested loops to iterate over
    simulation.
    """
    dt=max_temp
    iteration=0
    temp_next=np.copy(temp)
    (anumrows,anumcols)=temp.shape

    if verbose:
        ReportProgressText(temp,iteration,dt,"temp")
    while dt > min_temp_change and iteration < max_iteration:
        iteration+=1

        # Calculate temp at next time point. Each cell's temp is average
        # of its 4 neighbors.
        for i in range(1,anumrows-1):
            for j in range(1,anumcols-1):
                temp_next[i,j]=np.mean([temp[i-1,j],temp[i+1,j],temp[i,j-1],temp[i,j+1]])
        dt=np.amax(temp_next-temp)
        temp=np.copy(temp_next)
        if verbose and iteration % 10 == 0:
            ReportProgressText(temp,iteration,dt,"temp")

    ReportProgressText(temp[1:-1,1:-1],iteration,dt,"temp")
    return (dt,iteration)

def InitializeSimulation(anumrows,anumcols,max_temp):
    temp=np.zeros((anumrows,anumcols))
    # Set the right and bottom margins
    temp[:,-1]=np.linspace(0.0,max_temp,anumrows)
    temp[-1,:]=np.linspace(0.0,max_temp,anumcols)
    return temp

fig, ax = plt.subplots()

def main():

    parser=argparse.ArgumentParser(description="Laplace solver for nxn array")
    parser.add_argument("--dim",dest="dimension",type=int,default=10)
    parser.add_argument("--max_temp",dest="max_temp",type=int,default=100)
    parser.add_argument("--max_iterations",dest="max_iterations",type=int,default=1000)
    parser.add_argument("--method",dest="method",default="arrays",choices=["loops","arrays"])
    parser.add_argument("--verbose",dest="verbose",action="store_true")
    args=parser.parse_args()

    # Display precision:
    np.set_printoptions(precision=2)
    np.set_printoptions(edgeitems=10)
    np.set_printoptions(linewidth=150)

    # Simulation size:
    numcols=args.dimension
    numrows=args.dimension

    # Array size - 2 bigger because we need margins:
    anumcols=numcols+2
    anumrows=numrows+2

    max_temp=float(args.max_temp)
    min_temp_change=0.01

    iteration=0
    max_iteration=args.max_iterations

    temp = InitializeSimulation(anumrows,anumcols,max_temp)

    comm = MPI.COMM_WORLD

    if args.method == "loops":
        dt,iteration = LaplaceSolverNestedLoops(max_temp,max_iteration,min_temp_change,temp,args.verbose)
    else:
        dt,iteration = LaplaceSolverOverlappedArrays(comm,max_temp,max_iteration,min_temp_change,temp,args.verbose)

if __name__ == "__main__":
    main()
