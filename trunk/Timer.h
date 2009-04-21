#pragma once

#include <windows.h>

// brcc doesn't like this stuff so I guess I'll throw it here.

__int64 freq = 0;
__int64 clocks = 0;
__int64 start = 0;

void startTimer()
{
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&start);
}

void stopTimer()
{
	__int64 n;
	QueryPerformanceCounter((LARGE_INTEGER*)&n);
	n -= start;
	start = 0;
	clocks += n;
}

double getElapsedTime()
{
	return (double)clocks / (double)freq;
}