/*
//@HEADER
// *****************************************************************************
//
//  XtraPuLP: Xtreme-Scale Graph Partitioning using Label Propagation
//              Copyright (2016) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions?  Contact  George M. Slota   (gmslota@sandia.gov)
//                      Siva Rajamanickam (srajama@sandia.gov)
//                      Kamesh Madduri    (madduri@cse.psu.edu)
//
// *****************************************************************************
//@HEADER
*/


#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "io_pp.h"
#include "comms.h"
#include "util.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;


int load_graph_edges_32(char *input_filename, graph_gen_data_t *ggi,
                        bool offset_vids)
{
  if (debug) { printf("Task %d load_graph_edges_32() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

// #define EBIN
#ifndef EBIN
  // check file
  std::ifstream infile(input_filename);
  if (!infile) {
    throw_err("load_graph_edges_32() unable to open input file", procid);
  }

  /* METIS file format:
   * header is 2-4 numbers
   *    arg 1: # vertices
   *    arg 2: # edges. Unclear on how this should behave for undirected vs
   *        directed graphs; on undirected graphs, this number is half of the
   *        number of edges actually present in the file and 1/4 of the actual
   *        edges that need to be allocated.
   *    arg 3: 3-digit binary number, optional (for PuLP, should always be 010)
   *        0x4 = data contains vertex sizes (PuLP currently doesn't use this)
   *        0x2 = data contains vertex weights
   *        0x1 = data contains edge weights (PuLP currently doesn't use this)
   *    arg 4: number of weights for each vertex (required for PuLP)
   * each line i after this contains information about vertex i (starting at 1)
   *    first, all the weights of vertex i, as many as specified in the header
   *    then, all other vertices which have edges to vertex i
   */

  // header parsing
  std::istringstream ss;
  std::string line;
  uint64_t nverts_global;
  uint64_t nedges_global;
  std::string mode;
  uint32_t weights_per_vertex;
  getline(infile, line);
  ss.str(line);
  if (!(ss >> nverts_global >> nedges_global >> mode >> weights_per_vertex)) {
    throw_err("load_graph_edges_32() bad header format", procid);
  }
  if (mode != "010") {
    throw_err("load_graph_edges_32() bad fmt parameter: needs to be 010", procid);
  }

  // ASSUMING all graphs are undirected: XtraPuLP should double the global
  // number of edges because that's what it actually sees while reading the
  // file, basically translating this to the directed form of the undirected graph.
  nedges_global *= 2;

  // set global number of vertices and this process's local number of vertices
  ggi->n = nverts_global;

  ggi->n_local = ggi->n / (uint64_t)nprocs;
  if ((unsigned int) procid < ggi->n % nprocs) {
      // balancing effort; any spare vertices will get put into the first n%nprocs processes
      (ggi->n_local)++;
  }

// #define NEW_VERTEX_BALANCING
#ifdef NEW_VERTEX_BALANCING
  // For some reason, this new vertex balancing code is causing crashes later
  // in the program. Not yet sure why.

  // set n_offset, the formula is a bit complicated with how vertices are balanced
  if ((unsigned int) procid <= ggi->n % nprocs && ggi->n % nprocs > 0) {
      ggi->n_offset = ((ggi->n / nprocs) + 1) * procid;
  }
  else {
      ggi->n_offset = ggi->n - ((nprocs - procid) * (ggi->n / nprocs));
  }
#else
  ggi->n_offset = (uint64_t)procid * (ggi->n / (uint64_t)nprocs + 1);
  ggi->n_local = ggi->n / (uint64_t)nprocs + 1;
  if (procid == nprocs - 1 && !offset_vids)
    ggi->n_local = ggi->n - ggi->n_offset;
#endif

  // set global number of edges
  ggi->m = nedges_global;

  ggi->unscaled_vweights = NULL;
  if (weights_per_vertex > 0) {
    ggi->unscaled_vweights = (double*) calloc(weights_per_vertex * ggi->n_local, sizeof(double));
    if (ggi->unscaled_vweights == NULL) {
        throw_err("load_graph_edges(), unable to allocate vertex weight buffer", procid);
    }
    ggi->weights_per_vertex = weights_per_vertex;
  }

  /* bad hack: we don't know exactly how many edges are going to be read by the vertices of this
   * process, we only know the global number. We could either read through the
   * file once just to see how many edges there are in this set of vertices, or
   * allocate the maximum possible amount of space (the global number of edges).
   * In this case, allocate the maximum possible amount of space. */
  uint64_t* gen_edges_read = (uint64_t*) calloc(nedges_global * 2, sizeof(uint64_t));
  if (gen_edges_read == NULL) {
    throw_err("load_graph_edges(), unable to allocate edge buffer", procid);
  }

  // parse each vertex line
  uint64_t cur_vert = 0; // cur_vert = actual vertex index, counting from 0
  uint64_t edgectr = 0;  // edgectr = current edge index within gen_edges_read
  for (; getline(infile, line); cur_vert++) {
      if (cur_vert >= ggi->n) {
          // something is wrong with the header
          throw_err("load_graph_edges(), found more vertices than specified in the header", procid);
      }
      else if (cur_vert < ggi->n_offset) {
          continue;
      }
      else if (cur_vert >= ggi->n_offset + ggi->n_local) {
          break;
      }
      ss.clear();
      ss.str(line);
      for (uint64_t i = 0; i < weights_per_vertex; ++i) {
          // the nth vertex read by this process
          uint64_t weight_offset = (weights_per_vertex * (cur_vert - ggi->n_offset));
          if (!(ss >> ggi->unscaled_vweights[weight_offset + i])) {
              throw_err("load_graph_edges(), unable to read a vertex weight", procid);
          }
      }
      uint64_t endpoint;
      while (ss >> endpoint) {
          if (edgectr >= nedges_global) {
              printf("Process %d: trying to write %ld-th edge, was only aware of %ld globally\n",
                      procid, edgectr, nedges_global);
              throw_err("load_graph_edges(), writing in too many edges!", procid);
          }
          gen_edges_read[edgectr*2] = cur_vert;
          // note: needs to be endpoint-1 because the file counts vertexes
          // starting at 1 but the data structure (and cur_vert) count from 0.
          gen_edges_read[(edgectr*2) + 1] = endpoint-1;
          edgectr++;
      }
  }
  ggi->m_local_read = edgectr;

  // now fix gen_edges' allocation to use only the amount of edges used
  ggi->gen_edges = (uint64_t*) calloc(ggi->m_local_read*2, sizeof(uint64_t));
  if (ggi->gen_edges == NULL) {
    throw_err("load_graph_edges(), unable to allocate buffer", procid);
  }
  for (uint64_t i = 0; i < ggi->m_local_read*2; ++i) {
      ggi->gen_edges[i] = gen_edges_read[i];
  }

  /* // possibly use MPI_Allreduce to check this?
  if (edgectr != nedges_global) {
      printf("edges read = %ld - header = %ld", edgectr, nedges_global);
      throw_err("load_graph_edges_32() number of edges read doesn't match header", procid);
  }
  */
#else /* if EBIN */

  FILE *infp = fopen(input_filename, "rb");
  if(infp == NULL)
    throw_err("load_graph_edges_32() unable to open input file", procid);

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fseek(infp, 0L, SEEK_SET);

  uint64_t nedges_global = file_size/(2*sizeof(uint32_t));
  ggi->m = nedges_global;

  uint64_t read_offset_start = procid*2*sizeof(uint32_t)*(nedges_global/nprocs);
  uint64_t read_offset_end = (procid+1)*2*sizeof(uint32_t)*(nedges_global/nprocs);

  if (procid == nprocs - 1)
    read_offset_end = 2*sizeof(uint32_t)*nedges_global;

  uint64_t nedges = (read_offset_end - read_offset_start)/8;
  ggi->m_local_read = nedges;

  if (debug) {
    printf("Task %d, read_offset_start %ld, read_offset_end %ld, nedges_global %ld, nedges: %ld\n", procid, read_offset_start, read_offset_end, nedges_global, nedges);
  }

  uint32_t* gen_edges_read = (uint32_t*)malloc(2*nedges*sizeof(uint32_t));
  uint64_t* gen_edges = (uint64_t*)malloc(2*nedges*sizeof(uint64_t));
  if (gen_edges_read == NULL || gen_edges == NULL)
    throw_err("load_graph_edges(), unable to allocate buffer", procid);

  fseek(infp, read_offset_start, SEEK_SET);
  if (!fread(gen_edges_read, nedges, 2*sizeof(uint32_t), infp))
    throw_err("Error: load_graph_edges_32(), can't read input file");
  fclose(infp);

  for (uint64_t i = 0; i < nedges*2; ++i)
    gen_edges[i] = (uint64_t)gen_edges_read[i];

  free(gen_edges_read);
  ggi->gen_edges = gen_edges;

  uint64_t max_n = 0;
  for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    if (gen_edges[i] > max_n)
      max_n = gen_edges[i];

  uint64_t n_global;
  MPI_Allreduce(&max_n, &n_global, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  ggi->n = n_global+1;
  ggi->n_offset = (uint64_t)procid * (ggi->n / (uint64_t)nprocs + 1);
  ggi->n_local = ggi->n / (uint64_t)nprocs + 1;
  if (procid == nprocs - 1 && !offset_vids && !offset_vids)
    ggi->n_local = n_global - ggi->n_offset + 1;

#endif

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d read %lu edges, %9.6f (s)\n", procid, ggi->m_local_read, elt);
  }

  if (debug) {
  printf("Process %d: n:%ld, n_offset:%ld, n_local:%ld, m:%ld, m_l_r:%ld, m_l_e:%ld\n",
          procid, ggi->n, ggi->n_offset, ggi->n_local, ggi->m, ggi->m_local_read, ggi->m_local_edges);
  }

  if (offset_vids)
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    {
      // task id = vertex / processes; task = vertex % processes.
      uint64_t task_id = ggi->gen_edges[i] / (uint64_t)nprocs;
      uint64_t task = ggi->gen_edges[i] % (uint64_t)nprocs;
      // n / nprocs + 1 = vertices per process, so task_offset is the
      uint64_t task_offset = task * (ggi->n / (uint64_t)nprocs + 1);
      uint64_t new_vid = task_offset + task_id;
      new_vid = (new_vid >= ggi->n) ? (ggi->n - 1) : new_vid;
      ggi->gen_edges[i] = new_vid;
    }
  }

  if (verbose) {
    printf("Task %d, n %lu, n_offset %lu, n_local %lu\n",
           procid, ggi->n, ggi->n_offset, ggi->n_local);
  }

  if (debug) { printf("Task %d load_graph_edges() success\n", procid); }
  return 0;
}


int load_graph_edges_64(char *input_filename, graph_gen_data_t *ggi,
                        bool offset_vids)
{
  if (debug) { printf("Task %d load_graph_edges_64() start\n", procid); }

  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  FILE *infp = fopen(input_filename, "rb");
  if(infp == NULL)
    throw_err("load_graph_edges_64() unable to open input file", procid);

  fseek(infp, 0L, SEEK_END);
  uint64_t file_size = ftell(infp);
  fseek(infp, 0L, SEEK_SET);

  uint64_t nedges_global = file_size/(2*sizeof(uint32_t));
  ggi->m = nedges_global;

  uint64_t read_offset_start = procid*2*sizeof(uint32_t)*(nedges_global/nprocs);
  uint64_t read_offset_end = (procid+1)*2*sizeof(uint32_t)*(nedges_global/nprocs);

  if (procid == nprocs - 1)
    read_offset_end = 2*sizeof(uint32_t)*nedges_global;

  uint64_t nedges = (read_offset_end - read_offset_start)/8;
  ggi->m_local_read = nedges;

  if (debug) {
    printf("Task %d, read_offset_start %ld, read_offset_end %ld, nedges_global %ld, nedges: %ld\n", procid, read_offset_start, read_offset_end, nedges_global, nedges);
  }

  uint64_t* gen_edges = (uint64_t*)malloc(2*nedges*sizeof(uint64_t));
  if (gen_edges == NULL)
    throw_err("load_graph_edges(), unable to allocate buffer", procid);

  fseek(infp, read_offset_start, SEEK_SET);
  if (!fread(gen_edges, nedges, 2*sizeof(uint32_t), infp))
    throw_err("Error: load_graph_edges_64(), can't read input file");
  fclose(infp);

  ggi->gen_edges = gen_edges;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d read %lu edges, %9.6f (s)\n", procid, nedges, elt);
  }

  uint64_t max_n = 0;
  for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    if (gen_edges[i] > max_n)
      max_n = gen_edges[i];

  uint64_t n_global;
  MPI_Allreduce(&max_n, &n_global, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  ggi->n = n_global+1;
  ggi->n_offset = (uint64_t)procid * (ggi->n / (uint64_t)nprocs + 1);
  ggi->n_local = ggi->n / (uint64_t)nprocs + 1;
  if (procid == nprocs - 1 && !offset_vids && !offset_vids)
    ggi->n_local = n_global - ggi->n_offset + 1;


  if (offset_vids)
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < ggi->m_local_read*2; ++i)
    {
      uint64_t task_id = ggi->gen_edges[i] / (uint64_t)nprocs;
      uint64_t task = ggi->gen_edges[i] % (uint64_t)nprocs;
      uint64_t task_offset = task * (ggi->n / (uint64_t)nprocs + 1);
      uint64_t new_vid = task_offset + task_id;
      new_vid = (new_vid >= ggi->n) ? (ggi->n - 1) : new_vid;
      ggi->gen_edges[i] = new_vid;
    }
  }

  if (verbose) {
    printf("Task %d, n %lu, n_offset %lu, n_local %lu\n",
           procid, ggi->n, ggi->n_offset, ggi->n_local);
  }

  if (debug) { printf("Task %d load_graph_edges() success\n", procid); }
  return 0;
}


int exchange_edges(graph_gen_data_t *ggi, mpi_data_t* comm)
{
  if (debug) { printf("Task %d exchange_edges() start\n", procid); }
  double elt = 0.0;
  if (verbose) {
    MPI_Barrier(MPI_COMM_WORLD);
    elt = omp_get_wtime();
  }

  uint64_t* temp_sendcounts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  uint64_t* temp_recvcounts = (uint64_t*)malloc(nprocs*sizeof(uint64_t));
  for (int i = 0; i < nprocs; ++i)
  {
    temp_sendcounts[i] = 0;
    temp_recvcounts[i] = 0;
  }

  uint64_t n_per_rank = ggi->n / nprocs + 1;
  for (uint64_t i = 0; i < ggi->m_local_read*2; i+=2)
  {
    uint64_t vert1 = ggi->gen_edges[i];
    int32_t vert_task1 = (int32_t)(vert1 / n_per_rank);
    temp_sendcounts[vert_task1] += 2;

    uint64_t vert2 = ggi->gen_edges[i+1];
    int32_t vert_task2 = (int32_t)(vert2 / n_per_rank);
    temp_sendcounts[vert_task2] += 2;
  }

  MPI_Alltoall(temp_sendcounts, 1, MPI_UINT64_T,
               temp_recvcounts, 1, MPI_UINT64_T, MPI_COMM_WORLD);

  uint64_t total_recv = 0;
  uint64_t total_send = 0;
  for (int32_t i = 0; i < nprocs; ++i)
  {
    total_recv += temp_recvcounts[i];
    total_send += temp_sendcounts[i];
  }
  free(temp_sendcounts);
  free(temp_recvcounts);

  uint64_t* recvbuf = (uint64_t*)malloc(total_recv*sizeof(uint64_t));
  if (recvbuf == NULL)
  {
    fprintf(stderr, "Task %d Error: exchange_out_edges(), unable to allocate buffer\n", procid);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  uint64_t max_transfer = total_send > total_recv ? total_send : total_recv;
  uint64_t num_comms = max_transfer / (uint64_t)(MAX_SEND_SIZE/2) + 1;
  MPI_Allreduce(MPI_IN_PLACE, &num_comms, 1,
                MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  if (debug)
    printf("Task %d exchange_edges() num_comms %lu total_send %lu total_recv %lu\n", procid, num_comms, total_send, total_recv);

  uint64_t sum_recv = 0;
  for (uint64_t c = 0; c < num_comms; ++c)
  {
    uint64_t send_begin = (ggi->m_local_read * c) / num_comms;
    uint64_t send_end = (ggi->m_local_read * (c + 1)) / num_comms;
    if (c == (num_comms-1))
      send_end = ggi->m_local_read;

    for (int32_t i = 0; i < nprocs; ++i)
    {
      comm->sendcounts[i] = 0;
      comm->recvcounts[i] = 0;
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t vert1 = ggi->gen_edges[i*2];
      int32_t vert_task1 = (int32_t)(vert1 / n_per_rank);
      comm->sendcounts[vert_task1] += 2;

      uint64_t vert2 = ggi->gen_edges[i*2+1];
      int32_t vert_task2 = (int32_t)(vert2 / n_per_rank);
      comm->sendcounts[vert_task2] += 2;
    }

    MPI_Alltoall(comm->sendcounts, 1, MPI_INT32_T,
                 comm->recvcounts, 1, MPI_INT32_T, MPI_COMM_WORLD);

    comm->sdispls[0] = 0;
    comm->sdispls_cpy[0] = 0;
    comm->rdispls[0] = 0;
    for (int32_t i = 1; i < nprocs; ++i)
    {
      comm->sdispls[i] = comm->sdispls[i-1] + comm->sendcounts[i-1];
      comm->rdispls[i] = comm->rdispls[i-1] + comm->recvcounts[i-1];
      comm->sdispls_cpy[i] = comm->sdispls[i];
    }

    int32_t cur_send = comm->sdispls[nprocs-1] + comm->sendcounts[nprocs-1];
    int32_t cur_recv = comm->rdispls[nprocs-1] + comm->recvcounts[nprocs-1];
    uint64_t* sendbuf = (uint64_t*) malloc((uint64_t)cur_send*sizeof(uint64_t));
    if (sendbuf == NULL)
    {
      fprintf(stderr, "Task %d Error: exchange_out_edges(), unable to allocate comm buffers", procid);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (uint64_t i = send_begin; i < send_end; ++i)
    {
      uint64_t vert1 = ggi->gen_edges[2*i];
      uint64_t vert2 = ggi->gen_edges[2*i+1];
      int32_t vert_task1 = (int32_t)(vert1 / n_per_rank);
      int32_t vert_task2 = (int32_t)(vert2 / n_per_rank);

      sendbuf[comm->sdispls_cpy[vert_task1]++] = vert1;
      sendbuf[comm->sdispls_cpy[vert_task1]++] = vert2;
      sendbuf[comm->sdispls_cpy[vert_task2]++] = vert2;
      sendbuf[comm->sdispls_cpy[vert_task2]++] = vert1;
    }

    MPI_Alltoallv(sendbuf, comm->sendcounts, comm->sdispls, MPI_UINT64_T,
                  recvbuf+sum_recv, comm->recvcounts, comm->rdispls,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    sum_recv += cur_recv;
    free(sendbuf);
  }

  free(ggi->gen_edges);
  ggi->gen_edges = recvbuf;
  ggi->m_local_edges = total_recv / 2;

  if (verbose) {
    elt = omp_get_wtime() - elt;
    printf("Task %d exchange_out_edges() sent %lu, recv %lu, m_local_edges %lu, %9.6f (s)\n", procid, total_send, total_recv, ggi->m_local_edges, elt);
  }

  if (debug) { printf("Task %d exchange_out_edges() success\n", procid); }
  return 0;
}

/* Scale the unscaled vertex weights from ggi and place them in vertex_weights. */
void scale_weights(graph_gen_data_t* ggi, int scaling_method, int norming_method)
{
    // for shorthand
    const uint64_t wpv = ggi->weights_per_vertex;

    /* Scaling method values:
     * -1 = No scaling.
     *  0 = scale by reducing the weight of
     *  1 = unimplemented.
     */
    if(scaling_method > 0) {
        throw_err("scale_weights, bad scaling method");
    }
    else if (scaling_method == 0) {
        const double INT_EPSILON = 1e-5;

        // Compute the sum of each ordered weight. Only do the sum for this
        // process's vertices though, and then we will Allreduce it to get the
        // global sums of each ordered weight.
        // This step is analogous to the Zoltan aggregate_weights.
        double* local_weight_sum = (double*) calloc(wpv + 1, sizeof(double));
        double* global_weight_sum = (double*) calloc(wpv + 1, sizeof(double));

        // The reason for the +1 in the array size is so that we don't have to
        // call Allreduce twice; the last value is a sentinel for if there are
        // any non-integer weights.
        double* non_int_sentinel = local_weight_sum + wpv;
        *non_int_sentinel = 0;

        for (uint64_t j = 0; j < ggi->n_local; ++j) {
            for (uint64_t i = 0; i < ggi->weights_per_vertex; ++i) {
                double this_weight = ggi->unscaled_vweights[(j * wpv) + i];
                local_weight_sum[i] += this_weight;
                if (*non_int_sentinel == 0
                    && fabs(floor(this_weight + 0.5) - this_weight) > INT_EPSILON) {
                    *non_int_sentinel = 1;
                }
            }
        }

        // sum all local weight sums (plus sentinel) into the global buffer
        MPI_Allreduce(local_weight_sum, global_weight_sum, wpv+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // move the sentinel to global_weight_sum, as this represents the sum of
        // all the local sentinels
        non_int_sentinel = global_weight_sum + wpv;

        /* next step: scale the nth weight on each vertex, using three main inputs:
        * the nth global weight sum, the sentinel, and scaling_method. */
        const double MAX_ALLOWABLE_SUM = 1e9;
        /* Scale if any of the following are true:
        * 1) non-integer weights exist
        * 2) weights are too tiny
        * 3) weights are too large
        */
        double scale = 1.0;

        for (uint64_t i = 0; i < ggi->weights_per_vertex; ++i) {
            if (*non_int_sentinel > 0 || global_weight_sum[i] < INT_EPSILON
                || global_weight_sum[i] > MAX_ALLOWABLE_SUM) {
                if (scaling_method == 0) {
                    // not sure why this check exists...
                    if (global_weight_sum[i] > 0) {
                        scale = MAX_ALLOWABLE_SUM / global_weight_sum[i];
                    }
                }
                else if (scaling_method == 1) {
                    // TODO: this option is unimplemented!
                }
            }

            for (uint64_t j = 0; j < ggi->n_local; ++j) {
                const uint64_t index = (j * wpv) + i;
                ggi->unscaled_vweights[index] = ceil(ggi->unscaled_vweights[index] * scale);
            }
        }

        free(local_weight_sum);
        free(global_weight_sum);
    }
    /* if scaling_method < 0, do nothing here */

    /* If scaling has happened, "ggi->unscaled_vweights" now actually contains
     * scaled vweights, but to avoid having to create more arrays we just use
     * that. */

    /* Now norm the weights, reducing them to a single number.
     * TODO: do this in a way that doesn't necessarily reduce to a single
     * number, but a smaller number that's more efficient than re-partitioning
     * each time.
     * Values for the norming_method argument:
     *  0 or negative: do not norm
     *  1: 1-norm (sum)
     *  2: 2-norm (Euclidean)
     *  >2: infinity-norm (max)
     */
    if (norming_method <= 0) {
        /* Scaling but NO norming -> we allocate vertex_weights to be the same size
         * as it is now, with wpv weights per vertex, copy them in as int32_t's,
         * and be done.
         */
        ggi->vertex_weights = (int32_t*) calloc(wpv * ggi->n_local, sizeof(int32_t));
        /* Note: This used to be uint64_t, but was changed to match dist_graph_t vertex weights. */
        for (uint64_t i = 0; i < wpv * ggi->n_local; ++i) {
            ggi->vertex_weights[i] = (int32_t) ggi->unscaled_vweights[i];
        }

        free(ggi->unscaled_vweights);
        ggi->unscaled_vweights = NULL;
        return;
    }
    else {
        ggi->vertex_weights = (int32_t*) calloc(ggi->n_local, sizeof(int32_t));
        ggi->weights_per_vertex = 1;
    }

    double result = 0;
    for (uint64_t j = 0; j < ggi->n_local; ++j) {
        result = 0;
        for (uint64_t i = 0; i < wpv; ++i) {
            const uint64_t index = (j * wpv) + i;
            if (norming_method == 1) {
                result += ggi->unscaled_vweights[index];
            }
            else if (norming_method == 2) {
                // Possible overflow here? Is this a concern?
                // TODO: should this norm in particular be done when we're still dealing with doubles?
                result += (ggi->unscaled_vweights[index] * ggi->unscaled_vweights[index]);
            }
            else {
                if (ggi->unscaled_vweights[index] > result) {
                    result = ggi->unscaled_vweights[index];
                }
            }
        }
        if (norming_method == 2) {
            result = sqrt(result);
        }
        ggi->vertex_weights[j] = (int32_t) result;
    }

    // for (uint64_t j=0; j<ggi->n_local; ++j) {
    //     printf("%d %d\n", procid, ggi->vertex_weights[j]);
    // }
}
