// includes, system
//#include <cuda.h>
//#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
//#include "typedefs.h"

//C++ #defines
#define USE_8A_NBCUTOFF
#include "autocomm.h"
//#include "grid.h"
//#include "eval.h"
#include "constants.h"
//#include "trilinterp.h"
//#include "eintcal.h"
//#include "distdepdiel.h"
#include "cuda_wrapper.h"

// Dummy input
#include "dummyinput.h"

// other defines
#define NOSQRT
#define BLOCK_SIZE 128

// Constant cache
__constant__ float nonbondlist_c[NONBONDLISTS_SIZE];
__constant__ int nnb_array_c[NNB_ARRAY_SIZE];
__constant__ float strsol_fn_c[SOLFN_SIZE];

/**
 * eintcal GPU kernel, does eintcal energy calculations for each 
 * individual in the population.
 * @param num_individualsgpu number of individuals in population
 * @param natomsgpu number of atoms
 * @param penergiesgpu array of energies used to store individual's energy
 * @param nonbondlist (used in cpu eintcal)
 * @param tcoord (used in cpu eintcal)
 * @param B_include_1_4_interactions (used in cpu eintcal)
 * @param B_have_flexible_residues (used in cpu eintcal)
 * @param nnb_array (used in cpu eintcal)
 * @param Nb_group_energy (used in cpu eintcal)
 * @param stre_vdW_Hb (used in cpu eintcal)
 * @param strsol_fn (used in cpu eintcal)
 * @param strepsilon_fn (used in cpu eintcal)
 * @param strr_epsilon_fn (used in cpu eintcal)
 * @param b_comp_intermolgpu (used in cpu eintcal)
 * @param pfloat_arraygpu array of float variables used in cpu trilinterp
 * @param pint_arraygpu array of integer varibales used in cpu trilinterp
 */
__global__ void eintcal_kernel(unsigned int num_individualsgpu,
								int natomsgpu, 
								float *penergiesgpu, 
								float *nonbondlist, 
								float *tcoord, 
								int B_include_1_4_interactions, // Boole
								int B_have_flexible_residues, // Boole
								int *nnb_array, 
								float *Nb_group_energy, 
								float *stre_vdW_Hb, 
								float *strsol_fn, 
								float *strepsilon_fn, 
								float *strr_epsilon_fn,
								int b_comp_intermolgpu, // Boole
								float *pfloat_arraygpu,
								int *pint_arraygpu)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	

	if (idx < num_individualsgpu) {
		
		if (!pint_arraygpu[INTEVALFLAG * num_individualsgpu + idx])//!evalflagsgpu[idx])
		{
			
# ifndef NOSQRT
			float r = 0.0f;
			// float nbc = B_use_non_bond_cutoff[idx] ? NBC : 999;
			float nbc = (int)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC : 999; // Boole cast
# else
			// float nbc2 = B_use_non_bond_cutoff[idx] ? NBC2 : 999 * 999;
			float nbc2 = (int)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC2 : 999 * 999; // Boole cast
# endif
			float dx = 0.0f, dy = 0.0f, dz = 0.0f;
			float r2 = 0.0f;
			
			float total_e_internal = 0.0f;
			
			float e_elec = 0.0f;
			
			int inb = 0;
			int a1 = 0, a2 = 0;
			int t1 = 0, t2 = 0;
			int nonbond_type = 0;
			
			int index_1t_NEINT = 0;
			int index_1t_NDIEL = 0;
			int nb_group = 0;
			int inb_from = 0;
			int inb_to = 0;
			int nb_group_max = 1;
			
			if (B_have_flexible_residues) {
				nb_group_max = 3;
			}
			
			for (nb_group = 0; nb_group < nb_group_max; nb_group++)
			{
				if (nb_group == 0) {
					inb_from = 0;
				} else {
					inb_from = nnb_array[nb_group-1];
				}
				inb_to = nnb_array[nb_group];
				
				for (inb = inb_from; inb < inb_to; inb++)
				{
				
					float e_internal = 0.0f;
					float e_desolv = 0.0f;
					
					a1 = (int)nonbondlist[inb * 7 + 0];
					a2 = (int)nonbondlist[inb * 7 + 1];
					t1 = (int)nonbondlist[inb * 7 + 2];
					t2 = (int)nonbondlist[inb * 7 + 3];
					
					nonbond_type = (int)nonbondlist[inb * 7 + 4];
					float nb_desolv = nonbondlist[inb * 7 + 5];
					float q1q2 = nonbondlist[inb * 7 + 6];
					
					dx = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + X] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + X];
					dy = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Y] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Y];
					dz = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Z] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Z];
					
	#ifndef NOSQRT
					r = clamp(hypotenuse(dx,dy,dz), RMIN_ELEC);
					r2 = r*r;
					int index = Ang_to_index(r);
					
	#else
					r2 = sqhypotenuse(dx,dy,dz);
					r2 = clamp(r2, RMIN_ELEC2);
					int index = SqAng_to_index(r2);
	#endif
					
					index_1t_NEINT = BoundedNeint(index);
					index_1t_NDIEL = BoundedNdiel(index);
					
					if ((int)pint_arraygpu[INTINCELEC * num_individualsgpu + idx])//B_calcIntElec[idx]) // Boole cast
					{
						float r_dielectric = strr_epsilon_fn[index_1t_NDIEL];
						e_elec = q1q2 * r_dielectric;
						e_internal = e_elec;
					}
					
					if (r2 < nbc2) {
						e_desolv = strsol_fn[index_1t_NEINT] * nb_desolv;
						int myidx;
						if (B_include_1_4_interactions != 0 && nonbond_type == 4) {
							myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
							if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS) {
								// e_internal += scale_1_4[idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
								e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
							}
							else {
								// e_internal += scale_1_4[idx] * (stre_vdW_Hb[myidx] + e_desolv);
								e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx] + e_desolv);
							}
						} else {
							// fprintf(stderr," stre_vdW_Hb[%d][%d][%d] = %f\n", index_1t_NEINT, t2, t1, stre_vdW_Hb[index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1]);
							// e_internal += stre_vdW_Hb[index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1] + e_desolv;
							myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
							if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS) {
								e_internal += stre_vdW_Hb[myidx-1] + e_desolv;
								// fprintf(stderr,"NEINT = %d, index = %d, t2 = %d, t1 = %d\n", NEINT, index_1t_NEINT, t2, t1);
							}
							else {
								e_internal += stre_vdW_Hb[myidx] + e_desolv;
							}
					
						}
						
						
						
					}
					total_e_internal += e_internal;
				}
				
				if (nb_group == INTRA_LIGAND) 
				{
					Nb_group_energy[INTRA_LIGAND] = total_e_internal;
				} else if (nb_group == INTER) {
					Nb_group_energy[INTER] = total_e_internal - Nb_group_energy[INTRA_LIGAND];
				} else if (nb_group == INTRA_RECEPTOR) {
					Nb_group_energy[INTRA_RECEPTOR] = total_e_internal - Nb_group_energy[INTRA_LIGAND] - Nb_group_energy[INTER];
				}
			}
			
			if(b_comp_intermolgpu) {
				//energiesgpu[idx] += ((float)total_e_internal - (float)unboundinternalFEs[idx]);
				penergiesgpu[idx] += ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);//(float)unboundinternalFEs[idx]);
			}
			else {
				//energiesgpu[idx] = ((float)total_e_internal - (float)unboundinternalFEs[idx]);
				penergiesgpu[idx] = ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);//(float)unboundinternalFEs[idx]);
			}
		}
	}
}

__global__ void eintcal_kernel_shared(unsigned int num_individualsgpu,
								int natomsgpu, 
								float *penergiesgpu, 
								float *nonbondlist, 
								float *tcoord, 
								int B_include_1_4_interactions, // Boole
								int B_have_flexible_residues, // Boole
								int *nnb_array, 
								float *Nb_group_energy, 
								float *stre_vdW_Hb, 
								float *strsol_fn, 
								float *strepsilon_fn, 
								float *strr_epsilon_fn,
								int b_comp_intermolgpu, // Boole
								float *pfloat_arraygpu,
								int *pint_arraygpu)
{
	__shared__ float nonbondlist_s[NONBONDLISTS_SIZE];
	for (int i = threadIdx.x; i < NONBONDLISTS_SIZE; i += blockDim.x) {
		nonbondlist_s[i] = nonbondlist[i];
	}

	__syncthreads();

	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;

	if (idx + blockDim.x * blockIdx.x < num_individualsgpu) {
		
		if (!pint_arraygpu[INTEVALFLAG * num_individualsgpu + idx])//!evalflagsgpu[idx])
		{
			
# ifndef NOSQRT
			float r = 0.0f;
			// float nbc = B_use_non_bond_cutoff[idx] ? NBC : 999;
			float nbc = (int)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC : 999; // Boole cast
# else
			// float nbc2 = B_use_non_bond_cutoff[idx] ? NBC2 : 999 * 999;
			float nbc2 = (int)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC2 : 999 * 999; // Boole cast
# endif
			float dx = 0.0f, dy = 0.0f, dz = 0.0f;
			float r2 = 0.0f;
			
			float total_e_internal = 0.0f;
			
			float e_elec = 0.0f;
			
			int inb = 0;
			int a1 = 0, a2 = 0;
			int t1 = 0, t2 = 0;
			int nonbond_type = 0;
			
			int index_1t_NEINT = 0;
			int index_1t_NDIEL = 0;
			int nb_group = 0;
			int inb_from = 0;
			int inb_to = 0;
			int nb_group_max = 1;
			
			if (B_have_flexible_residues) {
				nb_group_max = 3;
			}
			
			for (nb_group = 0; nb_group < nb_group_max; nb_group++)
			{
				if (nb_group == 0) {
					inb_from = 0;
				} else {
					inb_from = nnb_array[nb_group-1];
				}
				inb_to = nnb_array[nb_group];
				
				for (inb = inb_from; inb < inb_to; inb++)
				{
				
					float e_internal = 0.0f;
					float e_desolv = 0.0f;
					
					a1 = (int)nonbondlist_s[inb * 7 + 0];
					a2 = (int)nonbondlist_s[inb * 7 + 1];
					t1 = (int)nonbondlist_s[inb * 7 + 2];
					t2 = (int)nonbondlist_s[inb * 7 + 3];
					
					nonbond_type = (int)nonbondlist_s[inb * 7 + 4];
					float nb_desolv = nonbondlist_s[inb * 7 + 5];
					float q1q2 = nonbondlist_s[inb * 7 + 6];
					
					dx = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + X] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + X];
					dy = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Y] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Y];
					dz = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Z] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Z];
					
	#ifndef NOSQRT
					r = clamp(hypotenuse(dx,dy,dz), RMIN_ELEC);
					r2 = r*r;
					int index = Ang_to_index(r);
					
	#else
					r2 = sqhypotenuse(dx,dy,dz);
					r2 = clamp(r2, RMIN_ELEC2);
					int index = SqAng_to_index(r2);
	#endif
					
					index_1t_NEINT = BoundedNeint(index);
					index_1t_NDIEL = BoundedNdiel(index);
					
					if ((int)pint_arraygpu[INTINCELEC * num_individualsgpu + idx])//B_calcIntElec[idx]) // Boole cast
					{
						float r_dielectric = strr_epsilon_fn[index_1t_NDIEL];
						e_elec = q1q2 * r_dielectric;
						e_internal = e_elec;
					}
					
					if (r2 < nbc2) {
						e_desolv = strsol_fn[index_1t_NEINT] * nb_desolv;
						int myidx;
						if (B_include_1_4_interactions != 0 && nonbond_type == 4) {
							myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
							if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS) {
								// e_internal += scale_1_4[idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
								e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
							}
							else {
								// e_internal += scale_1_4[idx] * (stre_vdW_Hb[myidx] + e_desolv);
								e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx] + e_desolv);
							}
						} else {
							// fprintf(stderr," stre_vdW_Hb[%d][%d][%d] = %f\n", index_1t_NEINT, t2, t1, stre_vdW_Hb[index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1]);
							// e_internal += stre_vdW_Hb[index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1] + e_desolv;
							myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
							if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS) {
								e_internal += stre_vdW_Hb[myidx-1] + e_desolv;
								// fprintf(stderr,"NEINT = %d, index = %d, t2 = %d, t1 = %d\n", NEINT, index_1t_NEINT, t2, t1);
							}
							else {
								e_internal += stre_vdW_Hb[myidx] + e_desolv;
							}
					
						}
						
						
					}
					total_e_internal += e_internal;
				}
				
				if (nb_group == INTRA_LIGAND) 
				{
					Nb_group_energy[INTRA_LIGAND] = total_e_internal;
				} else if (nb_group == INTER) {
					Nb_group_energy[INTER] = total_e_internal - Nb_group_energy[INTRA_LIGAND];
				} else if (nb_group == INTRA_RECEPTOR) {
					Nb_group_energy[INTRA_RECEPTOR] = total_e_internal - Nb_group_energy[INTRA_LIGAND] - Nb_group_energy[INTER];
				}
			}
			
			if(b_comp_intermolgpu) {
				//energiesgpu[idx] += ((float)total_e_internal - (float)unboundinternalFEs[idx]);
				penergiesgpu[idx] += ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);//(float)unboundinternalFEs[idx]);
			}
			else {
				//energiesgpu[idx] = ((float)total_e_internal - (float)unboundinternalFEs[idx]);
				penergiesgpu[idx] = ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);//(float)unboundinternalFEs[idx]);
			}
		}
	}
}

// const_1 uses more const memory arrays (nnb, nonbondlist, strsol)
__global__ void eintcal_kernel_const_1(unsigned int num_individualsgpu,
								int natomsgpu, 
								float *penergiesgpu, 
								float *tcoord, 
								int B_include_1_4_interactions, // Boole
								int B_have_flexible_residues, // Boole
								float *Nb_group_energy, 
								float *stre_vdW_Hb, 
								float *strepsilon_fn, 
								float *strr_epsilon_fn,
								int b_comp_intermolgpu, // Boole
								float *pfloat_arraygpu,
								int *pint_arraygpu)
{
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;

	if (idx + blockDim.x * blockIdx.x < num_individualsgpu) {
		
		if (!pint_arraygpu[INTEVALFLAG * num_individualsgpu + idx])//!evalflagsgpu[idx])
		{
			
# ifndef NOSQRT
			float r = 0.0f;
			// float nbc = B_use_non_bond_cutoff[idx] ? NBC : 999;
			float nbc = (int)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC : 999; // Boole cast
# else
			// float nbc2 = B_use_non_bond_cutoff[idx] ? NBC2 : 999 * 999;
			float nbc2 = (int)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC2 : 999 * 999; // Boole cast
# endif
			float dx = 0.0f, dy = 0.0f, dz = 0.0f;
			float r2 = 0.0f;
			
			float total_e_internal = 0.0f;
			
			float e_elec = 0.0f;
			
			int inb = 0;
			int a1 = 0, a2 = 0;
			int t1 = 0, t2 = 0;
			int nonbond_type = 0;
			
			int index_1t_NEINT = 0;
			int index_1t_NDIEL = 0;
			int nb_group = 0;
			int inb_from = 0;
			int inb_to = 0;
			int nb_group_max = 1;
			
			if (B_have_flexible_residues) {
				nb_group_max = 3;
			}
			
			for (nb_group = 0; nb_group < nb_group_max; nb_group++)
			{
				if (nb_group == 0) {
					inb_from = 0;
				} else {
					inb_from = nnb_array_c[nb_group-1];
				}
				inb_to = nnb_array_c[nb_group];
				
				for (inb = inb_from; inb < inb_to; inb++)
				{
				
					float e_internal = 0.0f;
					float e_desolv = 0.0f;
					
					a1 = (int)nonbondlist_c[inb * 7 + 0];
					a2 = (int)nonbondlist_c[inb * 7 + 1];
					t1 = (int)nonbondlist_c[inb * 7 + 2];
					t2 = (int)nonbondlist_c[inb * 7 + 3];
					
					nonbond_type = (int)nonbondlist_c[inb * 7 + 4];
					float nb_desolv = nonbondlist_c[inb * 7 + 5];
					float q1q2 = nonbondlist_c[inb * 7 + 6];
					
					dx = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + X] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + X];
					dy = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Y] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Y];
					dz = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Z] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Z];
					
	#ifndef NOSQRT
					r = clamp(hypotenuse(dx,dy,dz), RMIN_ELEC);
					r2 = r*r;
					int index = Ang_to_index(r);
					
	#else
					r2 = sqhypotenuse(dx,dy,dz);
					r2 = clamp(r2, RMIN_ELEC2);
					int index = SqAng_to_index(r2);
	#endif
					
					index_1t_NEINT = BoundedNeint(index);
					index_1t_NDIEL = BoundedNdiel(index);
					
					if ((int)pint_arraygpu[INTINCELEC * num_individualsgpu + idx])//B_calcIntElec[idx]) // Boole cast
					{
						float r_dielectric = strr_epsilon_fn[index_1t_NDIEL];
						e_elec = q1q2 * r_dielectric;
						e_internal = e_elec;
					}
					
					if (r2 < nbc2) {
						e_desolv = strsol_fn_c[index_1t_NEINT] * nb_desolv;
						int myidx;
						if (B_include_1_4_interactions != 0 && nonbond_type == 4) {
							myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
							if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS) {
								// e_internal += scale_1_4[idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
								e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
							}
							else {
								// e_internal += scale_1_4[idx] * (stre_vdW_Hb[myidx] + e_desolv);
								e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx] + e_desolv);
							}
						} else {
							// fprintf(stderr," stre_vdW_Hb[%d][%d][%d] = %f\n", index_1t_NEINT, t2, t1, stre_vdW_Hb[index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1]);
							// e_internal += stre_vdW_Hb[index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1] + e_desolv;
							myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
							if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS) {
								e_internal += stre_vdW_Hb[myidx-1] + e_desolv;
								// fprintf(stderr,"NEINT = %d, index = %d, t2 = %d, t1 = %d\n", NEINT, index_1t_NEINT, t2, t1);
							}
							else {
								e_internal += stre_vdW_Hb[myidx] + e_desolv;
							}
					
						}
						
						
					}
					total_e_internal += e_internal;
				}
				
				if (nb_group == INTRA_LIGAND) 
				{
					Nb_group_energy[INTRA_LIGAND] = total_e_internal;
				} else if (nb_group == INTER) {
					Nb_group_energy[INTER] = total_e_internal - Nb_group_energy[INTRA_LIGAND];
				} else if (nb_group == INTRA_RECEPTOR) {
					Nb_group_energy[INTRA_RECEPTOR] = total_e_internal - Nb_group_energy[INTRA_LIGAND] - Nb_group_energy[INTER];
				}
			}
			
			if(b_comp_intermolgpu) {
				//energiesgpu[idx] += ((float)total_e_internal - (float)unboundinternalFEs[idx]);
				penergiesgpu[idx] += ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);//(float)unboundinternalFEs[idx]);
			}
			else {
				//energiesgpu[idx] = ((float)total_e_internal - (float)unboundinternalFEs[idx]);
				penergiesgpu[idx] = ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);//(float)unboundinternalFEs[idx]);
			}
		}
	}
}

// const_2 uses fewer constant memory arrays (nnb and nonbondlists only)
__global__ void eintcal_kernel_const_2(unsigned int num_individualsgpu,
								int natomsgpu, 
								float *penergiesgpu, 
								float *tcoord, 
								int B_include_1_4_interactions, // Boole
								int B_have_flexible_residues, // Boole
								int *nnb_array, 
								float *Nb_group_energy, 
								float *stre_vdW_Hb, 
								float *strsol_fn, 
								float *strepsilon_fn, 
								float *strr_epsilon_fn,
								int b_comp_intermolgpu, // Boole
								float *pfloat_arraygpu,
								int *pint_arraygpu)
{
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = threadIdx.x;

	if (idx + blockDim.x * blockIdx.x < num_individualsgpu) {
		
		if (!pint_arraygpu[INTEVALFLAG * num_individualsgpu + idx])//!evalflagsgpu[idx])
		{
			
# ifndef NOSQRT
			float r = 0.0f;
			// float nbc = B_use_non_bond_cutoff[idx] ? NBC : 999;
			float nbc = (int)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC : 999; // Boole cast
# else
			// float nbc2 = B_use_non_bond_cutoff[idx] ? NBC2 : 999 * 999;
			float nbc2 = (int)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC2 : 999 * 999; // Boole cast
# endif
			float dx = 0.0f, dy = 0.0f, dz = 0.0f;
			float r2 = 0.0f;
			
			float total_e_internal = 0.0f;
			
			float e_elec = 0.0f;
			
			int inb = 0;
			int a1 = 0, a2 = 0;
			int t1 = 0, t2 = 0;
			int nonbond_type = 0;
			
			int index_1t_NEINT = 0;
			int index_1t_NDIEL = 0;
			int nb_group = 0;
			int inb_from = 0;
			int inb_to = 0;
			int nb_group_max = 1;
			
			if (B_have_flexible_residues) {
				nb_group_max = 3;
			}
			
			for (nb_group = 0; nb_group < nb_group_max; nb_group++)
			{
				if (nb_group == 0) {
					inb_from = 0;
				} else {
					inb_from = nnb_array_c[nb_group-1];
				}
				inb_to = nnb_array_c[nb_group];
				
				for (inb = inb_from; inb < inb_to; inb++)
				{
				
					float e_internal = 0.0f;
					float e_desolv = 0.0f;
					
					a1 = (int)nonbondlist_c[inb * 7 + 0];
					a2 = (int)nonbondlist_c[inb * 7 + 1];
					t1 = (int)nonbondlist_c[inb * 7 + 2];
					t2 = (int)nonbondlist_c[inb * 7 + 3];
					
					nonbond_type = (int)nonbondlist_c[inb * 7 + 4];
					float nb_desolv = nonbondlist_c[inb * 7 + 5];
					float q1q2 = nonbondlist_c[inb * 7 + 6];
					
					dx = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + X] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + X];
					dy = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Y] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Y];
					dz = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Z] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Z];
					
	#ifndef NOSQRT
					r = clamp(hypotenuse(dx,dy,dz), RMIN_ELEC);
					r2 = r*r;
					int index = Ang_to_index(r);
					
	#else
					r2 = sqhypotenuse(dx,dy,dz);
					r2 = clamp(r2, RMIN_ELEC2);
					int index = SqAng_to_index(r2);
	#endif
					
					index_1t_NEINT = BoundedNeint(index);
					index_1t_NDIEL = BoundedNdiel(index);
					
					if ((int)pint_arraygpu[INTINCELEC * num_individualsgpu + idx])//B_calcIntElec[idx]) // Boole cast
					{
						float r_dielectric = strr_epsilon_fn[index_1t_NDIEL];
						e_elec = q1q2 * r_dielectric;
						e_internal = e_elec;
					}
					
					if (r2 < nbc2) {
						e_desolv = strsol_fn[index_1t_NEINT] * nb_desolv;
						int myidx;
						if (B_include_1_4_interactions != 0 && nonbond_type == 4) {
							myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
							if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS) {
								// e_internal += scale_1_4[idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
								e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
							}
							else {
								// e_internal += scale_1_4[idx] * (stre_vdW_Hb[myidx] + e_desolv);
								e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx] + e_desolv);
							}
						} else {
							// fprintf(stderr," stre_vdW_Hb[%d][%d][%d] = %f\n", index_1t_NEINT, t2, t1, stre_vdW_Hb[index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1]);
							// e_internal += stre_vdW_Hb[index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1] + e_desolv;
							myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
							if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS) {
								e_internal += stre_vdW_Hb[myidx-1] + e_desolv;
								// fprintf(stderr,"NEINT = %d, index = %d, t2 = %d, t1 = %d\n", NEINT, index_1t_NEINT, t2, t1);
							}
							else {
								e_internal += stre_vdW_Hb[myidx] + e_desolv;
							}
					
						}
						
						
					}
					total_e_internal += e_internal;
				}
				
				if (nb_group == INTRA_LIGAND) 
				{
					Nb_group_energy[INTRA_LIGAND] = total_e_internal;
				} else if (nb_group == INTER) {
					Nb_group_energy[INTER] = total_e_internal - Nb_group_energy[INTRA_LIGAND];
				} else if (nb_group == INTRA_RECEPTOR) {
					Nb_group_energy[INTRA_RECEPTOR] = total_e_internal - Nb_group_energy[INTRA_LIGAND] - Nb_group_energy[INTER];
				}
			}
			
			if(b_comp_intermolgpu) {
				//energiesgpu[idx] += ((float)total_e_internal - (float)unboundinternalFEs[idx]);
				penergiesgpu[idx] += ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);//(float)unboundinternalFEs[idx]);
			}
			else {
				//energiesgpu[idx] = ((float)total_e_internal - (float)unboundinternalFEs[idx]);
				penergiesgpu[idx] = ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);//(float)unboundinternalFEs[idx]);
			}
		}
	}
}


int main()
{
	// @@ Declare variables
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;
	
	// CPU variables
	unsigned num_individuals = NUM_INDIVIDUALS;
	int		natomsgpu = CPUNATOMS;
	float	energiescpu[NUM_INDIVIDUALS];
	float	nonbondlist[NONBONDLISTS_SIZE];
	float	tcoord[CRDS_SIZE];
	int		B_include_1_4_interactions = 1;//INC14INTERACT; // Boole
	int		B_have_flexible_residues = 1;//HAVEFLEXRESIDUES; // Boole
	int		nnb_array[NNB_ARRAY_SIZE];
	float	nb_group_energy[NB_GROUP_ENERGY_SIZE];
	float	e_vdW_Hb[EVDWHB_SIZE];
	float	sol_fn[SOLFN_SIZE];
	float	epsilon_fn[EPSILONFN_SIZE];
	float	repsilon_fn[REPSILONFN_SIZE];
	int		b_comp_intermolgpu = 1;//B_COMP_INTERMOL; // Boole
	float	pfloat_array[FLOAT_ARRAY_SIZE];
	int 	pint_array[INT_ARRAY_SIZE];

	// Initialize arrays
	std::ifstream ifile("testinput.txt");
	// energies
	for (int i = 0; i < NUM_INDIVIDUALS; ++i) {
		if (!(ifile >> energiescpu[i])) {
			std::cerr << "File read failed at line 1 element " << i << std::endl;
			return -1;
		}
	}
	// nonbondlist
	for (int i = 0; i < NONBONDLISTS_SIZE; ++i) {
		if (!(ifile >> nonbondlist[i])) {
			std::cerr << "File read failed at line 2 element " << i << std::endl;
			return -1;
		}
	}
	// tcoord
	for (int i = 0; i < CRDS_SIZE; ++i) {
		if (!(ifile >> tcoord[i])) {
			std::cerr << "File read failed at line 3 element " << i << std::endl;
			return -1;
		}
	}
	// nnbarray
	for (int i = 0; i < NNB_ARRAY_SIZE; ++i) {
		if (!(ifile >> nnb_array[i])) {
			std::cerr << "File read failed at line 4 element " << i << std::endl;
			return -1;
		}
	}
	// nb_group_energy
	for (int i = 0; i < NB_GROUP_ENERGY_SIZE; ++i) {
		if (!(ifile >> nb_group_energy[i])) {
			std::cerr << "File read failed at line 5 element " << i << std::endl;
			return -1;
		}
	}
	// e_vdW_Hb
	for (int i = 0; i < EVDWHB_SIZE; ++i) {
		if (!(ifile >> e_vdW_Hb[i])) {
			std::cerr << "File read failed at line 6 element " << i << std::endl;
			return -1;
		}
	}
	// sol_fn
	for (int i = 0; i < SOLFN_SIZE; ++i) {
		if (!(ifile >> sol_fn[i])) {
			std::cerr << "File read failed at line 7 element " << i << std::endl;
			return -1;
		}
	}
	// epsilon_fn
	for (int i = 0; i < EPSILONFN_SIZE; ++i) {
		if (!(ifile >> epsilon_fn[i])) {
			std::cerr << "File read failed at line 8 element " << i << std::endl;
			return -1;
		}
	}
	// repsilon_fn
	for (int i = 0; i < REPSILONFN_SIZE; ++i) {
		if (!(ifile >> repsilon_fn[i])) {
			std::cerr << "File read failed at line 9 element " << i << std::endl;
			return -1;
		}
	}
	// pfloat_array
	for (int i = 0; i < FLOAT_ARRAY_SIZE; ++i) {
		if (!(ifile >> pfloat_array[i])) {
			std::cerr << "File read failed at line 10 element " << i << std::endl;
			return -1;
		}
	}
	// pint_array
	for (int i = 0; i < INT_ARRAY_SIZE; ++i) {
		if (!(ifile >> pint_array[i])) {
			std::cerr << "File read failed at line 11 element " << i << std::endl;
			return -1;
		}
	}
	ifile.close();
	
	// GPU Variables
	float	*energiesgpu;
	float	*nonbondlistgpu;
	float	*tcoordgpu;
	int		*nnb_arraygpu;
	float	*Nb_group_energygpu;
	float	*stre_vdW_Hb;
	float	*strsol_fn;
	float	*strepsilon_fn;
	float	*strr_epsilon_fn;
	float	*pfloat_arraygpu;
	int		*pint_arraygpu;
	
	//@@ Allocate GPU memory here, add cudaEventRecord
	cudaEventRecord(start, 0);
	cudaMalloc((void**) &energiesgpu, NUM_INDIVIDUALS * sizeof(float));
	cudaMalloc((void**) &nonbondlistgpu, NONBONDLISTS_SIZE * sizeof(float));
	cudaMalloc((void**) &tcoordgpu, CRDS_SIZE * sizeof(float));
	cudaMalloc((void**) &nnb_arraygpu, NNB_ARRAY_SIZE * sizeof(int));
	cudaMalloc((void**) &Nb_group_energygpu, NB_GROUP_ENERGY_SIZE * sizeof(float));
	cudaMalloc((void**) &stre_vdW_Hb, EVDWHB_SIZE * sizeof(float));
	cudaMalloc((void**) &strsol_fn, SOLFN_SIZE * sizeof(float));
	cudaMalloc((void**) &strepsilon_fn, EPSILONFN_SIZE * sizeof(float));
	cudaMalloc((void**) &strr_epsilon_fn, REPSILONFN_SIZE * sizeof(float));
	cudaMalloc((void**) &pfloat_arraygpu, FLOAT_ARRAY_SIZE * sizeof(float));
	cudaMalloc((void**) &pint_arraygpu, INT_ARRAY_SIZE * sizeof(int));
	cudaEventRecord(stop, 0);
	std::cerr << "Requested " << NUM_INDIVIDUALS * sizeof(float) + NONBONDLISTS_SIZE * sizeof(float) + CRDS_SIZE * sizeof(float) + NNB_ARRAY_SIZE * sizeof(int) + NB_GROUP_ENERGY_SIZE * sizeof(float) + EVDWHB_SIZE * sizeof(float) + SOLFN_SIZE * sizeof(float) + EPSILONFN_SIZE * sizeof(float) + REPSILONFN_SIZE * sizeof(float) + FLOAT_ARRAY_SIZE * sizeof(float) + INT_ARRAY_SIZE * sizeof(int) << " bytes\n";

	// Synchronize
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Memory allocation took %f milliseconds\n", time);

	//@@ Copy memory to the GPU here
	cudaEventRecord(start, 0);
	cudaMemcpy(energiesgpu, energiescpu, NUM_INDIVIDUALS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(nonbondlistgpu, nonbondlist, NONBONDLISTS_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(tcoordgpu, tcoord, CRDS_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(nnb_arraygpu, nnb_array, NNB_ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Nb_group_energygpu, nb_group_energy, NB_GROUP_ENERGY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(stre_vdW_Hb, e_vdW_Hb, EVDWHB_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(strsol_fn, sol_fn, SOLFN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(strepsilon_fn, epsilon_fn, EPSILONFN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(strr_epsilon_fn, repsilon_fn, REPSILONFN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pfloat_arraygpu, pfloat_array, FLOAT_ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pint_arraygpu, pint_array, INT_ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	// Constant memory
	cudaMemcpyToSymbol(nonbondlist_c, nonbondlist, NONBONDLISTS_SIZE * sizeof(float));
	cudaMemcpyToSymbol(nnb_array_c, nnb_array, NNB_ARRAY_SIZE * sizeof(int));
	cudaMemcpyToSymbol(strsol_fn_c, sol_fn, SOLFN_SIZE * sizeof(float));
	cudaEventRecord(stop, 0);

	// Synchronize
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Memory transfer (host->device) took %f milliseconds\n", time);

	//@@ Initialize the grid and block dimensions here
	dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((num_individuals - 1) / BLOCK_SIZE + 1);
	
	//@@ Launch the original GPU Kernel here
	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; ++i) {
		eintcal_kernel<<<gridSize, blockSize>>>(num_individuals,
												natomsgpu, 
												energiesgpu, 
												nonbondlistgpu, 
												tcoordgpu, 
												B_include_1_4_interactions, 
												B_have_flexible_residues, 
												nnb_arraygpu, 
												Nb_group_energygpu, 
												stre_vdW_Hb, 
												strsol_fn, 
												strepsilon_fn, 
												strr_epsilon_fn,
												b_comp_intermolgpu,
												pfloat_arraygpu,
												pint_arraygpu);
	}
	cudaEventRecord(stop, 0);

	// Synchronize
	//std::cerr << cudaEventSynchronize(stop) << std::endl;
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);
	printf("Original kernel execution took %f milliseconds\n", time / 100.);
	
	//@@ Launch the shared mem GPU Kernel here
	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; ++i) {
		eintcal_kernel_shared<<<gridSize, blockSize>>>(num_individuals,
												natomsgpu, 
												energiesgpu, 
												nonbondlistgpu, 
												tcoordgpu, 
												B_include_1_4_interactions, 
												B_have_flexible_residues, 
												nnb_arraygpu, 
												Nb_group_energygpu, 
												stre_vdW_Hb, 
												strsol_fn, 
												strepsilon_fn, 
												strr_epsilon_fn,
												b_comp_intermolgpu,
												pfloat_arraygpu,
												pint_arraygpu);
	}
	cudaEventRecord(stop, 0);

	// Synchronize
	//std::cerr << cudaEventSynchronize(stop) << std::endl;
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);
	printf("Shared kernel execution took %f milliseconds\n", time / 100.);
	
	//@@ Launch the first const Kernel here
	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; ++i) {
		eintcal_kernel_const_1<<<gridSize, blockSize>>>(num_individuals,
												natomsgpu, 
												energiesgpu, 
												tcoordgpu, 
												B_include_1_4_interactions, 
												B_have_flexible_residues, 
												Nb_group_energygpu, 
												stre_vdW_Hb, 
												strepsilon_fn, 
												strr_epsilon_fn,
												b_comp_intermolgpu,
												pfloat_arraygpu,
												pint_arraygpu);
	}
	cudaEventRecord(stop, 0);

	// Synchronize
	//std::cerr << cudaEventSynchronize(stop) << std::endl;
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);
	printf("Const 1 kernel execution took %f milliseconds\n", time / 100.);
	
	
	cudaEventRecord(start, 0);
	for (int i = 0; i < 100; ++i) {
		eintcal_kernel_const_2<<<gridSize, blockSize>>>(num_individuals,
												natomsgpu, 
												energiesgpu, 
												tcoordgpu, 
												B_include_1_4_interactions, 
												B_have_flexible_residues, 
												nnb_arraygpu, 
												Nb_group_energygpu, 
												stre_vdW_Hb, 
												strsol_fn, 
												strepsilon_fn, 
												strr_epsilon_fn,
												b_comp_intermolgpu,
												pfloat_arraygpu,
												pint_arraygpu);
	}
	cudaEventRecord(stop, 0);

	// Synchronize
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&time, start, stop);
	printf("Constant 2 kernel execution took %f milliseconds\n", time / 100.);
	
	//@@ Copy the GPU memory back to the CPU here, add cudaEventRecord
	cudaEventRecord(start, 0);
	cudaMemcpy(energiescpu, energiesgpu, sizeof(float) * num_individuals, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Memory transfer (device->host) took %f milliseconds\n", time);

	//@@ Free the GPU memory here, add cudaEventRecord
	cudaEventRecord(start, 0);
	cudaFree(energiesgpu);
	cudaFree(nonbondlistgpu);
	cudaFree(tcoordgpu);
	cudaFree(nnb_arraygpu);
	cudaFree(Nb_group_energygpu);
	cudaFree(stre_vdW_Hb);
	cudaFree(strsol_fn);
	cudaFree(strepsilon_fn);
	cudaFree(strr_epsilon_fn);
	cudaFree(pfloat_arraygpu);
	cudaFree(pint_arraygpu);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Memory deallocation took %f milliseconds\n", time);
	//@@ end cudaEventRecord

	FILE *ofile = fopen("testout.txt", "w");
	fprintf(ofile, "{");
	for (int i = 0; i != NUM_INDIVIDUALS; ++i) {
		fprintf(ofile, "%f, ", energiescpu[i]);
	}
	fprintf(ofile, "}\n");
	fclose(ofile);

	return 0;
}
