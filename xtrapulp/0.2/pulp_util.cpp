
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
#include <fstream>
#include <cstring>
#include <assert.h>

#include "xtrapulp.h"
#include "pulp_util.h"
#include "pulp_data.h"

extern int procid, nprocs;
extern bool verbose, debug, verify;


int part_eval(dist_graph_t* g, pulp_data_t* pulp)
{
  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    pulp->part_sizes[i] = 0;
    pulp->part_edge_sizes[i] = 0;
    pulp->part_cut_sizes[i] = 0;
  }
  pulp->cut_size = 0;
  pulp->max_cut = 0;

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp->local_parts[vert_index];
    ++pulp->part_sizes[part];

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    pulp->part_edge_sizes[part] += (int64_t)out_degree;
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp->local_parts[out_index];
      if (part_out != part)
      {
        ++pulp->part_cut_sizes[part];
        ++pulp->cut_size;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, pulp->part_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp->cut_size /= 2;

  uint64_t global_ghost = g->n_ghost;
  uint64_t max_ghost = 0;
  MPI_Allreduce(&global_ghost, &max_ghost, 1,
    MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_ghost, 1,
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  double ghost_balance =
    (double)max_ghost / ((double)global_ghost / (double)pulp->num_parts);

  int64_t max_v_size = 0;
  int32_t max_v_part = -1;
  int64_t max_e_size = 0;
  int32_t max_e_part = -1;
  int64_t max_c_size = 0;
  int32_t max_c_part = -1;
  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    if (max_v_size < pulp->part_sizes[i])
    {
      max_v_size = pulp->part_sizes[i];
      max_v_part = i;
    }
    if (max_e_size < pulp->part_edge_sizes[i])
    {
      max_e_size = pulp->part_edge_sizes[i];
      max_e_part = i;
    }
    if (max_c_size < pulp->part_cut_sizes[i])
    {
      max_c_size = pulp->part_cut_sizes[i];
      max_c_part = i;
    }
  }

  pulp->max_v = (double)max_v_size / ((double)g->n / (double)pulp->num_parts);
  pulp->max_e = (double)max_e_size / ((double)g->m*2.0 / (double)pulp->num_parts);
  pulp->max_c = (double)max_c_size / ((double)pulp->cut_size / (double)pulp->num_parts);
  pulp->max_cut = max_c_size;

  if (procid == 0)
  {
    printf("---------------------------------------------------------\n");
    printf("EdgeCut: %li, MaxPartCut: %li\nVertexBalance: %2.3lf (%d, %li), EdgeBalance: %2.3lf (%d, %li)\nCutBalance: %2.3lf (%d, %li), GhostBalance: %2.3lf (%li)\n",
      pulp->cut_size, pulp->max_cut,
      pulp->max_v, max_v_part, max_v_size,
      pulp->max_e, max_e_part, max_e_size,
      pulp->max_c, max_c_part, max_c_size,
      ghost_balance, max_ghost);
    printf("---------------------------------------------------------\n");
    for (int32_t i = 0; i < pulp->num_parts; ++i)
      printf("Part: %d, VertSize: %li, EdgeSize: %li, Cut: %li\n",
      i, pulp->part_sizes[i], pulp->part_edge_sizes[i], pulp->part_cut_sizes[i]);
    printf("---------------------------------------------------------\n");
  }

  return 0;
}

int part_eval(dist_graph_t* g, int32_t* parts, int32_t num_parts)
{
  pulp_data_t pulp;
  init_pulp_data(g, &pulp, num_parts);
  memcpy(parts, pulp.local_parts, g->n_local*sizeof(int32_t));

  for (int32_t i = 0; i < pulp.num_parts; ++i)
  {
    pulp.part_sizes[i] = 0;
    pulp.part_edge_sizes[i] = 0;
    pulp.part_cut_sizes[i] = 0;
  }
  pulp.cut_size = 0;
  pulp.max_cut = 0;

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp.local_parts[vert_index];
    ++pulp.part_sizes[part];

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    pulp.part_edge_sizes[part] += (int64_t)out_degree;
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp.local_parts[out_index];
      if (part_out != part)
      {
        ++pulp.part_cut_sizes[part];
        ++pulp.cut_size;
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, pulp.part_sizes, pulp.num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp.part_edge_sizes, pulp.num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp.part_cut_sizes, pulp.num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp.cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp.cut_size /= 2;

  uint64_t global_ghost = g->n_ghost;
  uint64_t max_ghost = 0;
  MPI_Allreduce(&global_ghost, &max_ghost, 1,
    MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_ghost, 1,
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  double ghost_balance =
    (double)max_ghost / ((double)global_ghost / (double)pulp.num_parts);

  int64_t max_v_size = 0;
  int32_t max_v_part = -1;
  int64_t max_e_size = 0;
  int32_t max_e_part = -1;
  int64_t max_c_size = 0;
  int32_t max_c_part = -1;
  for (int32_t i = 0; i < pulp.num_parts; ++i)
  {
    if (max_v_size < pulp.part_sizes[i])
    {
      max_v_size = pulp.part_sizes[i];
      max_v_part = i;
    }
    if (max_e_size < pulp.part_edge_sizes[i])
    {
      max_e_size = pulp.part_edge_sizes[i];
      max_e_part = i;
    }
    if (max_c_size < pulp.part_cut_sizes[i])
    {
      max_c_size = pulp.part_cut_sizes[i];
      max_c_part = i;
    }
  }

  pulp.max_v = (double)max_v_size / ((double)g->n / (double)pulp.num_parts);
  pulp.max_e = (double)max_e_size / ((double)g->m*2.0 / (double)pulp.num_parts);
  pulp.max_c = (double)max_c_size / ((double)pulp.cut_size / (double)pulp.num_parts);
  pulp.max_cut = max_c_size;

  if (procid == 0)
  {
    printf("EVAL ec: %li, vb: %2.3lf (%d, %li), eb: %2.3lf (%d, %li), cb: %2.3lf (%d, %li), gb: %2.3lf (%li)\n",
      pulp.cut_size,
      pulp.max_v, max_v_part, max_v_size,
      pulp.max_e, max_e_part, max_e_size,
      pulp.max_c, max_c_part, max_c_size,
      ghost_balance, max_ghost);

    for (int32_t i = 0; i < pulp.num_parts; ++i)
      printf("p: %d, v: %li, e: %li, cut: %li\n",
      i, pulp.part_sizes[i], pulp.part_edge_sizes[i], pulp.part_cut_sizes[i]);
  }

  clear_pulp_data(&pulp);

  return 0;
}


int part_eval_weighted(dist_graph_t* g, pulp_data_t* pulp)
{
  bool has_vwgts = (g->original_vweights != NULL);
  bool has_ewgts = (g->edge_weights != NULL);
  const uint64_t owpv = g->original_weights_per_vertex;

  /* This array uses owpv as its first dimension to cut down on the number of
   * MPI_Allreduce calls; we generally expect owpv to be small, whereas the
   * number of parts could be small (in which case it's irrelevant) or quite
   * large.
   * The reason for using num_parts+2 is so we can later store the maximum and
   * sum in the last slot. */
#define WEIGHT_MAX_INDEX (pulp->num_parts)
#define WEIGHT_SUM_INDEX (pulp->num_parts + 1)
  double part_weighted_sizes[owpv][pulp->num_parts + 2];

  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    pulp->part_sizes[i] = 0;
    pulp->part_edge_sizes[i] = 0;
    pulp->part_cut_sizes[i] = 0;
    for (uint64_t j = 0; j < owpv; ++j)
        part_weighted_sizes[j][i] = 0;
  }
  pulp->cut_size = 0;
  pulp->max_cut = 0;

  if (has_vwgts) {
    for (uint64_t j = 0; j < owpv; ++j)
    {
      part_weighted_sizes[j][WEIGHT_MAX_INDEX] = -INFINITY;
      part_weighted_sizes[j][WEIGHT_SUM_INDEX] = 0;
    }
  }

  for (uint64_t i = 0; i < g->n_local; ++i)
  {
    uint64_t vert_index = i;
    int32_t part = pulp->local_parts[vert_index];

    /* part_sizes used to either be the total number of vertices or the total
     * weight of vertices in one part, but since we are now tracking weights
     * separately, it just always holds the total vertices. */
    ++pulp->part_sizes[part];

    if (has_vwgts) {
      // FIXME: g->original_vweights are doubles, we crudely cast them to ints
      // here.
      for (uint64_t j = 0; j < owpv; ++j)
      {
        double plusweight = g->original_vweights[(i * owpv) + j];
        part_weighted_sizes[j][part] += plusweight;
      }
    }

    uint64_t out_degree = out_degree(g, vert_index);
    uint64_t* outs = out_vertices(g, vert_index);
    int32_t* weights = out_weights(g, vert_index);
    pulp->part_edge_sizes[part] += (int64_t)out_degree;
    for (uint64_t j = 0; j < out_degree; ++j)
    {
      uint64_t out_index = outs[j];
      int32_t part_out = pulp->local_parts[out_index];
      if (part_out != part)
      {
        if (has_ewgts)
        {
          pulp->part_cut_sizes[part] += weights[j];
          pulp->cut_size += weights[j];
        }
        else
        {
          ++pulp->part_cut_sizes[part];
          ++pulp->cut_size;
        }
      }
    }
  }

  for (uint64_t weight = 0; weight < owpv; ++weight) {
    MPI_Allreduce(MPI_IN_PLACE, part_weighted_sizes[weight], pulp->num_parts,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  MPI_Allreduce(MPI_IN_PLACE, pulp->part_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_edge_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pulp->part_cut_sizes, pulp->num_parts,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pulp->cut_size, 1,
    MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  pulp->cut_size /= 2;

  int64_t part_size_sum = 0;
  int64_t part_edge_size_sum = 0;
  int64_t part_cut_size_sum = 0;

  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    part_size_sum += pulp->part_sizes[i];
    part_edge_size_sum += pulp->part_edge_sizes[i];
    part_cut_size_sum += pulp->part_cut_sizes[i];

    if (pulp->part_cut_sizes[i] > pulp->max_cut)
      pulp->max_cut = pulp->part_cut_sizes[i];
  }

  uint64_t global_ghost = g->n_ghost;
  uint64_t max_ghost = 0;
  MPI_Allreduce(&global_ghost, &max_ghost, 1,
    MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &global_ghost, 1,
    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  double ghost_balance =
    (double)max_ghost / ((double)global_ghost / (double)pulp->num_parts);

  int64_t max_v_size = 0;
  int32_t max_v_part = -1;
  int64_t max_e_size = 0;
  int32_t max_e_part = -1;
  int64_t max_c_size = 0;
  int32_t max_c_part = -1;
  for (int32_t i = 0; i < pulp->num_parts; ++i)
  {
    if (max_v_size < pulp->part_sizes[i])
    {
      max_v_size = pulp->part_sizes[i];
      max_v_part = i;
    }
    if (max_e_size < pulp->part_edge_sizes[i])
    {
      max_e_size = pulp->part_edge_sizes[i];
      max_e_part = i;
    }
    if (max_c_size < pulp->part_cut_sizes[i])
    {
      max_c_size = pulp->part_cut_sizes[i];
      max_c_part = i;
    }

    for (uint64_t wgt = 0; wgt < owpv; ++wgt) {
      double& weight_max = part_weighted_sizes[wgt][WEIGHT_MAX_INDEX];
      weight_max = std::max(weight_max, part_weighted_sizes[wgt][i]);
      part_weighted_sizes[wgt][WEIGHT_SUM_INDEX] += part_weighted_sizes[wgt][i];
    }
  }

  pulp->max_v = (double)max_v_size / ((double)part_size_sum / (double)pulp->num_parts);
  pulp->max_e = (double)max_e_size / ((double)g->m*2.0 / (double)pulp->num_parts);
  pulp->max_c = (double)max_c_size / ((double)pulp->cut_size / (double)pulp->num_parts);
  pulp->max_cut = max_c_size;

  if (procid == 0)
  {
#ifdef OLD_EVAL_STATEMENTS
    printf("EVAL ec: %li, vb: %2.3lf (%d, %li), eb: %2.3lf (%d, %li), cb: %2.3lf (%d, %li), gb: %2.3lf (%li)\n",
      pulp->cut_size,
      pulp->max_v, max_v_part, max_v_size,
      pulp->max_e, max_e_part, max_e_size,
      pulp->max_c, max_c_part, max_c_size,
      ghost_balance, max_ghost);

    for (int32_t i = 0; i < pulp->num_parts; ++i)
      printf("p: %d, v: %li, e: %li, cut: %li\n",
      i, pulp->part_sizes[i], pulp->part_edge_sizes[i], pulp->part_cut_sizes[i]);
#else
    printf("---------------------------------------------------------\n");
    printf("EdgeCut: %li,", pulp->cut_size);
    for (uint64_t i = 0; i < owpv; ++i) {
      /* Same calculation as max_v above, but considering the sum of weights
       * rather than vertex count. */
      double max_weight_imbalance = part_weighted_sizes[i][WEIGHT_MAX_INDEX] / (part_weighted_sizes[i][WEIGHT_SUM_INDEX] / (double) pulp->num_parts);
      printf(" WeightBalance %ld: %2.3lf", i, max_weight_imbalance);
    }
    printf("\n");
#endif
  }

  return 0;
}

int output_parts(const char* filename, dist_graph_t* g, int32_t* parts)
{
  output_parts(filename, g, parts, false);

  return 0;
}


int output_parts(const char* filename, dist_graph_t* g,
                 int32_t* parts, bool offset_vids)
{
  if (verbose) printf("Task %d writing parts to %s\n", procid, filename);

  int32_t* global_parts = (int32_t*)malloc(g->n*sizeof(int32_t));

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_parts[i] = -1;

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_local; ++i)
    if (offset_vids)
    {
      uint64_t task_id = g->local_unmap[i] - g->n_offset;
      uint64_t task = (uint64_t)procid;
      uint64_t global_id = task_id * (uint64_t)nprocs + task;
      if (global_id < g->n)
        global_parts[global_id] = parts[i];
    }
    else {
      global_parts[g->local_unmap[i]] = parts[i];
    }

  if (procid == 0)
    MPI_Reduce(MPI_IN_PLACE, global_parts, (int32_t)g->n,
      MPI_INT32_T, MPI_MAX, 0, MPI_COMM_WORLD);
  else
    MPI_Reduce(global_parts, global_parts, (int32_t)g->n,
      MPI_INT32_T, MPI_MAX, 0, MPI_COMM_WORLD);

  if (procid == 0)
  {
    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_parts[i] == -1)
        {
          printf("Part error: %lu not assigned\n", i);
          global_parts[i] = 0;
        }

    std::ofstream outfile;
    outfile.open(filename);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile << global_parts[i] << std::endl;

    outfile.close();
  }

  free(global_parts);

  if (verbose) printf("Task %d done writing parts\n", procid);

  return 0;
}

int read_parts(const char* filename, dist_graph_t* g,
               pulp_data_t* pulp, bool offset_vids)
{
  if (verbose) printf("Task %d reading in parts from %s\n", procid, filename);

  int32_t* global_parts = (int32_t*)malloc(g->n*sizeof(int32_t));

#pragma omp parallel for
  for (uint64_t i = 0; i < g->n; ++i)
    global_parts[i] = -1;
#pragma omp parallel for
  for (uint64_t i = 0; i < g->n_total; ++i)
    pulp->local_parts[i] = -1;


  if (procid == 0)
  {
    std::ifstream outfile;
    outfile.open(filename);

    for (uint64_t i = 0; i < g->n; ++i)
      outfile >> global_parts[i];

    outfile.close();

    if (debug)
      for (uint64_t i = 0; i < g->n; ++i)
        if (global_parts[i] == -1)
        {
          printf("Part error: %lu not assigned\n", i);
          global_parts[i] = 0;
        }
  }

  MPI_Bcast(global_parts, (int32_t)g->n, MPI_INT32_T, 0, MPI_COMM_WORLD);

  if (offset_vids)
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < g->n_local; ++i)
    {
      uint64_t task = (uint64_t)procid;
      uint64_t task_id = g->local_unmap[i] - g->n_offset;
      uint64_t global_id = task_id * (uint64_t)nprocs + task;
      if (global_id < g->n)
        pulp->local_parts[i] = global_parts[global_id];
    }
#pragma omp parallel for
    for (uint64_t i = 0; i < g->n_ghost; ++i)
    {
      uint64_t task = (uint64_t)g->ghost_tasks[i];
      uint64_t task_id = g->ghost_unmap[i] - task*(g->n/(uint64_t)nprocs + 1);
      uint64_t global_id = task_id * (uint64_t)nprocs + task;
      if (global_id < g->n)
        pulp->local_parts[i + g->n_local] = global_parts[global_id];
    }
    if (debug)
      for (uint64_t i = 0; i < g->n_total; ++i)
        if (pulp->local_parts[i] == -1)
        {
          printf("Part error: %lu not assigned\n", i);
          pulp->local_parts[i] = 0;
        }
  }
  else
  {
#pragma omp parallel for
    for (uint64_t i = 0; i < g->n_local; ++i)
      pulp->local_parts[i] = global_parts[g->local_unmap[i]];
#pragma omp parallel for
    for (uint64_t i = 0; i < g->n_ghost; ++i)
      pulp->local_parts[i + g->n_local] = global_parts[g->ghost_unmap[i]];
  }

  free(global_parts);

  if (verbose) printf("Task %d done reading in parts\n", procid);

  return 0;
}
