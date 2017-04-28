/////////////////////////////////////////////////////////////////
// Converted GS Algorithm for MRI Image Reconstruction to CUDA //
//															   //
// Authors: NGUYEN Hong Quan								   //
//			TRAN Nguyen Phuong Trinh						   //
//															   //
// Emails: nguyenhongquan_eeit13@hotmail.com				   //
//		   trinhtran2151995@gmail.com						   //
//															   //
// Date: 8th October, 2016									   //
/////////////////////////////////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <arrayfire.h>
#include <af/cuda.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <cufft.h>
#include <string>
#include <cublas_v2.h>
////////////////////////////////////////////////
using namespace std;
////////////////////////////////////////////////
const char kMaskFilePath[] = "E:\\MRI\\workspace\\Datas\\kmask1.txt";
const char dataFilePath[] = "E:\\MRI\\workspace\\Datas\\underSampledData1.txt";
const char IrefPath[] = "E:\\MRI\\workspace\\Datas\\irefData.txt";
string HroGSPath = "E:\\MRI\\workspace\\Results\\HroGS";
////////////////////////////////////////////////
const int IterNumCG = 250;
int NFrame, NSlice, NY, NX, NSamples;
float **cpu_data, **cpu_rhogs;
float *cpu_Iref;
int *cpu_kMaskIndex;
////////////////////////////////////////////////
af::array Apk, alpha, rsnew, p, r, c_gs, rsold, gpu_Iref, gpu_kMaskIndex, gpu_FrameDatas, idx, gpu_rhogs;
////////////////////////////////////////////////
void readData();
void writeResult_RhoGS();
////////////////////////////////////////////////
int main()
{
	af::setDevice(0);
	af_info();
	auto FrameCGFunction = [](){
		auto fftshift3D = [](af::array in){
			return af::shift(in, (int)in.dims(0) / 2, (int)in.dims(1) / 2, (int)in.dims(2) / 2);
		};

		auto ifftshift3D = [](af::array in){
			return af::shift(in, ((int)in.dims(0) + 1) / 2, ((int)in.dims(1) + 1) / 2, ((int)in.dims(2) + 1) / 2);
		};

		for (int id = 0; id < NFrame; id++)
		{
			r = gpu_FrameDatas(af::span, af::span, af::span, id);
			r = af::flat(r);
			p = r;
			rsold = af::dot(r, r, AF_MAT_CONJ, AF_MAT_NONE);
			c_gs = af::constant(0, r.dims(0), c32);
			for (int i = 0; i < IterNumCG; i++)
			{
				Apk = af::moddims(p, NSlice, NY, NX, 1);
				Apk = ifftshift3D(af::fft3(fftshift3D(gpu_Iref*ifftshift3D(af::ifft3(fftshift3D(Apk)))))); Apk.eval();
				Apk = af::flat(Apk)*idx;
				alpha = af::real((rsold) / (af::dot(p, Apk, AF_MAT_CONJ, AF_MAT_NONE))); alpha.eval();
				c_gs += af::tile(alpha, p.dims(0))*p; c_gs.eval();
				r -= af::tile(alpha, Apk.dims(0))*Apk; r.eval();
				rsnew = af::dot(r, r, AF_MAT_CONJ, AF_MAT_NONE);
				p = r + af::tile(rsnew / rsold, p.dims(0))*p; p.eval();
				rsold = rsnew;
				af::sync();
			}
			c_gs *= idx;
			c_gs = af::moddims(c_gs, NSlice, NY, NX, 1);
			gpu_rhogs = af::abs(gpu_Iref*ifftshift3D(af::ifft3(fftshift3D(c_gs))));
			gpu_rhogs.host(cpu_rhogs[id]);
			af::sync();
		}
	};
	auto start = chrono::high_resolution_clock::now();
	readData();
	auto stop = chrono::high_resolution_clock::now();
	cout << "readData() take " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << ".ms" << endl;
	start = chrono::high_resolution_clock::now();
	FrameCGFunction();
	stop = chrono::high_resolution_clock::now();
	cout << "process() take " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << ".ms" << endl;
	start = chrono::high_resolution_clock::now();
	writeResult_RhoGS();
	stop = chrono::high_resolution_clock::now();
	cout << "writeResult() take " << chrono::duration_cast<chrono::milliseconds>(stop - start).count() << ".ms" << endl;
	cudaDeviceSynchronize();
	cudaDeviceReset();
	free(cpu_kMaskIndex); free(cpu_Iref);
	for (int i = 0; i < NFrame; i++)
	{
		free(cpu_data[i]);
		free(cpu_rhogs[i]);
	}
	free(cpu_data);	free(cpu_rhogs);
	return 0;
}
////////////////////////////////////////////////
void readData()
{
	FILE *dataFile, *kMaskFile, *IrefFile;
	dataFile = fopen(dataFilePath, "r");
	kMaskFile = fopen(kMaskFilePath, "r");
	IrefFile = fopen(IrefPath, "r");

	//read kMask
	fscanf(kMaskFile, "%d", &NSamples);
	cpu_kMaskIndex = (int*)calloc(NSamples, sizeof(int));
	for (int i = 0; i < NSamples; i++)
	{
		fscanf(kMaskFile, "%d\n", &cpu_kMaskIndex[i]);
		cpu_kMaskIndex[i] -= 1;
	}
	gpu_kMaskIndex = af::array(NSamples, cpu_kMaskIndex, afHost);

	//read data
	fscanf(dataFile, "%d %d %d %d\n", &NFrame, &NSlice, &NY, &NX);
	cpu_data = (float**)malloc(sizeof(float*)*NFrame);
	cpu_rhogs = (float**)malloc(sizeof(float*)*NFrame);
	for (int i = 0; i < NFrame; i++)
	{
		cpu_data[i] = (float*)calloc(NSlice*NY*NX * 2, sizeof(float));
		cpu_rhogs[i]= (float*)calloc(NSlice*NY*NX, sizeof(float));
	}
	for (int i = 0; i < NFrame; i++)
	{
		for (int j = 0; j < NSamples; j++)
		{
			fscanf(dataFile, "%f %f\n", &(cpu_data[i][cpu_kMaskIndex[j] * 2]),
				&(cpu_data[i][cpu_kMaskIndex[j] * 2 + 1]));
		}
	}

	// copy host data -> GPU
	gpu_FrameDatas = af::array(NSlice, NY, NX, NFrame, c32);
	for (int i = 0; i < NFrame; i++)
	{
		gpu_FrameDatas(af::span, af::span, af::span, i) = af::array(NSlice, NY, NX, (af::cfloat*) cpu_data[i]);
	}

	//read Iref
	cpu_Iref = (float*)calloc(NSlice*NY*NX, sizeof(float));
	for (int i = 0; i < NSlice*NX*NY; i++)
	{
		fscanf(IrefFile, "%f", &cpu_Iref[i]);
	}
	gpu_Iref = af::array(NSlice, NY, NX, cpu_Iref, afHost);

	idx = af::constant(0, NSlice, NY, NX, c32);
	idx = af::flat(idx);
	idx(gpu_kMaskIndex) = 1;

	// close data files
	fclose(dataFile); fclose(kMaskFile); fclose(IrefFile);
}
//////////////////////////////////////////////////
void writeResult_RhoGS()
{
	for (int i = 0; i < NFrame; i++)
	{
		FILE* resultFile = fopen((HroGSPath + to_string(i) + string(".txt")).c_str(), "w");
		for (int j = 0; j < NSlice*NY*NX; j++)
		{
			fprintf(resultFile, "%f\n", cpu_rhogs[i][j]);
		}
		fclose(resultFile);
	}
}
/////////////////////////////////////////////////



