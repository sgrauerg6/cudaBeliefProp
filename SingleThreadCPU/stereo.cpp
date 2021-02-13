/*
 Copyright (C) 2006 Pedro Felzenszwalb

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 */

#include "stereo.h"

#ifdef _WIN32

__declspec(dllexport) RunBpStereoSet<float>* __cdecl createRunBpStereoCPUSingleThreadFloat()
{
	return new RunBpStereoCPUSingleThread<float>();
}

__declspec(dllexport) RunBpStereoSet<double>* __cdecl createRunBpStereoCPUSingleThreadDouble()
{
	return new RunBpStereoCPUSingleThread<double>();
}

__declspec(dllexport) RunBpStereoSet<short>* __cdecl createRunBpStereoCPUSingleThreadShort()
{
	return new RunBpStereoCPUSingleThread<short>();
}

#endif //_WIN32
