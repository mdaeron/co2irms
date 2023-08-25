#! /usr/bin/env python3

__version__ = 2.1

import numpy as np

def ratios_to_deltas(R45, R46,
	R13_VPDB = 0.01118,  # (Chang & Li, 1990)
	R18_VSMOW = 0.0020052,  # (Baertschi, 1976)
	R17_VSMOW = 0.00038475,  # (Assonov & Brenninkmeijer, 2003, rescaled to R13_VPDB)
	LAMBDA_17 = 0.528,  # (Barkan & Luz, 2005)
	D17O = 0 # in permil
	):

	from scipy.optimize import fsolve
	
	R45 = np.asarray(R45)
	R46 = np.asarray(R46)
	if R45.shape != R46.shape:
		raise ValueError('R45 and R46 must both be floats or both be arrays of the same shape.')

	def f(R18):
		K = np.exp(D17O/1e3) * R17_VSMOW / R18_VSMOW**LAMBDA_17
		return (-3 * K**2 * R18**(2*LAMBDA_17) + 2 * K * R45 * R18**LAMBDA_17 + 2 * R18 - R46)
	
	R18 = fsolve(f, R46/R18_VSMOW/2, xtol = 1e-16)
	R17 = np.exp(D17O/1e3) * R17_VSMOW * (R18 / R18_VSMOW) ** LAMBDA_17
	R13 = R45 - 2 * R17

	d13C_VPDB = (R13 / R13_VPDB - 1)*1e3
	d18O_VSMOW = (R18 / R18_VSMOW - 1)*1e3

	return (d13C_VPDB, d18O_VSMOW)

def deltas_to_ratios(d13C_VPDB, d18O_VSMOW,
	R13_VPDB = 0.01118,  # (Chang & Li, 1990)
	R18_VSMOW = 0.0020052,  # (Baertschi, 1976)
	R17_VSMOW = 0.00038475,  # (Assonov & Brenninkmeijer, 2003, rescaled to R13_VPDB)
	LAMBDA_17 = 0.528,  # (Barkan & Luz, 2005)
	D17O = 0 # in permil
	):

	d13C_VPDB = np.asarray(d13C_VPDB)
	d18O_VSMOW = np.asarray(d18O_VSMOW)
	if d13C_VPDB.shape != d18O_VSMOW.shape:
		raise ValueError('d13C_VPDB and d18O_VSMOW must both be floats or both be arrays of the same shape.')

	R13 = R13_VPDB * (1 + d13C_VPDB/1e3)
	R18 = R18_VSMOW * (1 + d18O_VSMOW/1e3)
	R17 = np.exp(D17O/1e3) * R17_VSMOW * (1 + d18O_VSMOW/1e3)**LAMBDA_17
	
	R45 = 2 * R17 + R13
	R46 = 2 * R18 + 2 * R17 * R13 + R17**2
	
	return(R45, R46)


def sanitize(x):
	return x.replace('-', '_').replace('.', '_')

def standardize(
	data,
	anchors,
	alpha18_acid = 1.008129,
	constraints = {},
	):

	import lmfit, warnings
	from scipy.stats import t as tstudent

# 	import scipy.linalg, lmfit
# 	from numpy import log, array, exp, cov, ix_, sqrt
# 	from scipy.stats import f as ssf

	out = {'data': [r.copy() for r in data]}

	sessions = sorted({r['Session'] for r in data})
	samples = sorted({r['Sample'] for r in data})

	fitparams = lmfit.Parameters()
	for s in sessions:
		fitparams.add('d45_scaling_'+sanitize(s), value = 1.)
		fitparams.add('d46_scaling_'+sanitize(s), value = 1.)
		fitparams.add('d13C_VPDB_of_wg_'+sanitize(s), value = 0.)
		fitparams.add('d18O_VSMOW_of_wg_'+sanitize(s), value = 0.)
	for s in samples:
		fitparams.add('d13C_VPDB_of_'+sanitize(s), value = 0.)
		fitparams.add('d18O_VSMOW_of_'+sanitize(s), value = 0.)

	for a in anchors:
		if 'd13C_VPDB' in anchors[a]:
			fitparams[f'd13C_VPDB_of_'+sanitize(a)].expr = str(anchors[a]['d13C_VPDB'])
		if 'd18O_VPDB' in anchors[a]:
			fitparams[f'd18O_VSMOW_of_'+sanitize(a)].expr = str(
				(1000 + anchors[a]['d18O_VPDB']) * alpha18_acid * 1.03092 - 1000
				)

	for p in fitparams:
		if p in constraints:
			fitparams[p].expr = constraints[p]

	delta_wg = np.array([[r['d45'], r['d46']] for r in data]) # delta_wg.shape = (N, 2)

	def residuals(p, sigma13, sigma18):
		Rwg = np.array([
			deltas_to_ratios(p[f'd13C_VPDB_of_wg_'+sanitize(r['Session'])], p[f'd18O_VSMOW_of_wg_'+sanitize(r['Session'])])
			for r in data
			]) # Rwg.shape = (N, 2)
		scaling = np.array([
			[p[f'd45_scaling_'+sanitize(r['Session'])], p[f'd46_scaling_'+sanitize(r['Session'])]]
			for r in data
			]) # scaling.shape = (N, 2)
		R = Rwg * (1 + delta_wg / scaling / 1000)
		d13, d18 = ratios_to_deltas(*R.T)

		d13true = np.array([p[f'd13C_VPDB_of_'+sanitize(r['Sample'])] for r in data])
		d18true = np.array([p[f'd18O_VSMOW_of_'+sanitize(r['Sample'])] for r in data])

		return np.hstack((
			(d13 - d13true) / sigma13,
			(d18 - d18true) / sigma18,
			))

	N = len(data)
	Nf13, Nf18 = N, N
	for p in fitparams:
		if fitparams[p].expr is not None:
			if p.startswith('d13') or p.startswith('d45'):
				Nf13 -= 1
			if p.startswith('d18') or p.startswith('d46'):
				Nf18 -= 1

	sigma13, sigma18 = 1., 1.

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		for k in range(2):
			fitresult = lmfit.minimize(residuals, fitparams, method = 'least_squares', scale_covar = True, args = (sigma13, sigma18))
			sigma13 *= ((fitresult.residual[:N]**2).sum() / Nf13)**.5
			sigma18 *= ((fitresult.residual[-N:]**2).sum() / Nf18)**.5

	out['bestfit'] = fitresult
	out['fitreport'] = lmfit.fit_report(fitresult)
	
	out['sigma_d13C'] = sigma13
	out['sigma_d18O'] = sigma18
	out['Nf_d13C'] = Nf13
	out['Nf_d18O'] = Nf18
	out['t95_d13C'] = tstudent.ppf(1 - 0.05/2, Nf13)
	out['t95_d18O'] = tstudent.ppf(1 - 0.05/2, Nf13)
	
	out['sigma_d13C_VPDB'] = sigma13
	out['sigma_d18O_VSMOW'] = sigma18
	out['sigma_d18O_VPDB'] = sigma18 / 1.03092 / alpha18_acid

	p = {_: fitresult.params[_].value for _ in fitresult.params}

	for k,r in enumerate(out['data']):
		r['d13C_VPDB_residual'] = fitresult.residual[k] * sigma13
		r['d18O_VSMOW_residual'] = fitresult.residual[k+N] * sigma18
		r['d18O_VPDB_residual'] = r['d18O_VSMOW_residual'] / 1.03092 / alpha18_acid

		r['d13C_VPDB'] = r['d13C_VPDB_residual'] + p['d13C_VPDB_of_'+sanitize(r['Sample'])]
		r['d18O_VSMOW'] = r['d18O_VSMOW_residual'] + p['d18O_VSMOW_of_'+sanitize(r['Sample'])]
		r['d18O_VPDB'] = (1000 + r[f'd18O_VSMOW']) / 1.03092 / alpha18_acid - 1000

	out['sessions'] = {}
	for s in sessions:
		_s = sanitize(s)
		out['sessions'][s] = {}

		out['sessions'][s]['N'] = len([r for r in data if r['Session'] == s])
		out['sessions'][s]['Na_d13C'] = len([
			r for r in data
			if r['Session'] == s
			and r['Sample'] in anchors
			and 'd13C_VPDB' in anchors[r['Sample']]
			])
		out['sessions'][s]['Na_d18O'] = len([
			r for r in data
			if r['Session'] == s
			and r['Sample'] in anchors
			and 'd18O_VPDB' in anchors[r['Sample']]
			])
		out['sessions'][s]['Nu_d13C'] = out['sessions'][s]['N'] - out['sessions'][s]['Na_d13C']
		out['sessions'][s]['Nu_d18O'] = out['sessions'][s]['N'] - out['sessions'][s]['Na_d18O']
		out['sessions'][s]['data'] = [r for r in out['data'] if r['Session'] == s]

		out['sessions'][s]['d45_scaling'] = fitresult.params[f'd45_scaling_'+_s].value
		out['sessions'][s]['d46_scaling'] = fitresult.params[f'd46_scaling_'+_s].value
		out['sessions'][s]['d13C_VPDB_of_wg'] = fitresult.params['d13C_VPDB_of_wg_'+_s].value
		out['sessions'][s]['d18O_VSMOW_of_wg'] = fitresult.params['d18O_VSMOW_of_wg_'+_s].value

		out['sessions'][s]['SE_d45_scaling'] = fitresult.params[f'd45_scaling_'+_s].stderr
		out['sessions'][s]['SE_d46_scaling'] = fitresult.params[f'd46_scaling_'+_s].stderr
		out['sessions'][s]['SE_d13C_VPDB_of_wg'] = fitresult.params['d13C_VPDB_of_wg_'+_s].stderr
		out['sessions'][s]['SE_d18O_VSMOW_of_wg'] = fitresult.params['d18O_VSMOW_of_wg_'+_s].stderr

		out['sessions'][s][f'RMSE_d13C_VPDB']   = (np.array([r[f'd13C_VPDB_residual'] for r in out['sessions'][s]['data']])**2).mean()**.5
		out['sessions'][s][f'RMSE_d18O_VSMOW']   = (np.array([r[f'd18O_VSMOW_residual'] for r in out['sessions'][s]['data']])**2).mean()**.5
		out['sessions'][s][f'RMSE_d18O_VPDB']   = (np.array([r[f'd18O_VPDB_residual'] for r in out['sessions'][s]['data']])**2).mean()**.5

	out['samples'] = {}
	for s in samples:
		out['samples'][s] = {}

		out['samples'][s]['N'] = len([r for r in data if r['Sample'] == s])
		out['samples'][s]['data'] = [r for r in out['data'] if r['Sample'] == s]
# 
	for s in samples:
		_s = sanitize(s)

		out['samples'][s]['d13C_VPDB'] = fitresult.params['d13C_VPDB_of_'+_s].value
		out['samples'][s]['SD_d13C_VPDB'] = np.array([r['d13C_VPDB_residual'] for r in out['samples'][s]['data']]).std(ddof = 1)
		if fitresult.params['d13C_VPDB_of_'+_s].stderr:
			out['samples'][s]['SE_d13C_VPDB'] = fitresult.params['d13C_VPDB_of_'+_s].stderr
			out['samples'][s]['95CL_d13C_VPDB'] = out['samples'][s]['SE_d13C_VPDB'] * out['t95_d13C']

		out['samples'][s]['d18O_VSMOW'] = fitresult.params['d18O_VSMOW_of_'+_s].value
		out['samples'][s]['SD_d18O_VSMOW'] = np.array([r['d18O_VSMOW_residual'] for r in out['samples'][s]['data']]).std(ddof = 1)
		if fitresult.params['d18O_VSMOW_of_'+_s].stderr:
			out['samples'][s]['SE_d18O_VSMOW'] = fitresult.params['d18O_VSMOW_of_'+_s].stderr
			out['samples'][s]['95CL_d18O_VSMOW'] = out['samples'][s]['SE_d18O_VSMOW'] * out['t95_d18O']
 
		out['samples'][s]['d18O_VPDB'] = (1000 + out['samples'][s]['d18O_VSMOW']) / 1.03092 / alpha18_acid - 1000
		out['samples'][s]['SD_d18O_VPDB'] = np.array([r['d18O_VPDB_residual'] for r in out['samples'][s]['data']]).std(ddof = 1)
		if fitresult.params['d18O_VSMOW_of_'+_s].stderr:
			out['samples'][s]['SE_d18O_VPDB'] = out['samples'][s]['SE_d18O_VSMOW'] / 1.03092 / alpha18_acid
			out['samples'][s]['95CL_d18O_VPDB'] = out['samples'][s]['SE_d18O_VPDB'] * out['t95_d18O']

	csv = f'Session,N,Na_d13C,Nu_d13C,Na_d18O,Nu_d18O,d45_scaling,SE_d45_scaling,d46_scaling,SE_d46_scaling,d13C_VPDB_of_wg,SE_d13C_VPDB_of_wg,d18O_VSMOW_of_wg,SE_d18O_VSMOW_of_wg,RMSE_d13C_VPDB,RMSE_d18O_VPDB'
	for s in sessions:
		csv += f'\n{s}'
		csv += f',{out["sessions"][s]["N"]}'
		csv += f',{out["sessions"][s]["Na_d13C"]}'
		csv += f',{out["sessions"][s]["Nu_d13C"]}'
		csv += f',{out["sessions"][s]["Na_d18O"]}'
		csv += f',{out["sessions"][s]["Nu_d18O"]}'
		csv += f',{out["sessions"][s]["d45_scaling"]:.4f}'
		csv += f',{out["sessions"][s]["SE_d45_scaling"]:.4f}'
		csv += f',{out["sessions"][s]["d46_scaling"]:.4f}'
		csv += f',{out["sessions"][s]["SE_d46_scaling"]:.4f}'
		csv += f',{out["sessions"][s]["d13C_VPDB_of_wg"]:.3f}'
		csv += f',{out["sessions"][s]["SE_d13C_VPDB_of_wg"]:.3f}'
		csv += f',{out["sessions"][s]["d18O_VSMOW_of_wg"]:.3f}'
		csv += f',{out["sessions"][s]["SE_d18O_VSMOW_of_wg"]:.3f}'
		csv += f',{out["sessions"][s]["RMSE_d13C_VPDB"]:.3f}'
		csv += f',{out["sessions"][s]["RMSE_d18O_VPDB"]:.3f}'
	out['csv_sessions'] = csv

	csv = f'Sample,N,d13C_VPDB,SE_d13C_VPDB,95CL_d13C_VPDB,SD_d13C_VPDB,d18O_VPDB,SE_d18O_VPDB,95CL_d18O_VPDB,SD_d18O_VPDB'
	for s in samples:
		csv += f'\n{s}'
		csv += f',{out["samples"][s]["N"]}'
		csv += f',{out["samples"][s]["d13C_VPDB"]:.3f}'
		try:
			csv += f',{out["samples"][s]["SE_d13C_VPDB"]:.3f}'
			csv += f',{out["samples"][s]["95CL_d13C_VPDB"]:.3f}'
		except KeyError:
			csv += ',,'
		csv += f',{out["samples"][s]["SD_d13C_VPDB"]:.3f}'
		csv += f',{out["samples"][s]["d18O_VPDB"]:.3f}'
		try:
			csv += f',{out["samples"][s]["SE_d18O_VPDB"]:.3f}'
			csv += f',{out["samples"][s]["95CL_d18O_VPDB"]:.3f}'
		except KeyError:
			csv += ',,'
		csv += f',{out["samples"][s]["SD_d18O_VPDB"]:.3f}'
	out['csv_samples'] = csv

	csv = f'UID,Session,Sample,d45,d46,d13C_VPDB,d18O_VPDB,d13C_VPDB_residual,d18O_VPDB_residual'
	for r in out['data']:
		csv += f'\n{r["UID"]}'
		csv += f',{r["Session"]}'
		csv += f',{r["Sample"]}'
		csv += f',{r["d45"]},{r["d46"]}'
		csv += f',{r["d13C_VPDB"]:.3f},{r["d18O_VPDB"]:.3f}'
		csv += f',{r["d13C_VPDB_residual"]:.3f},{r["d18O_VPDB_residual"]:.3f}'
	out['csv_analyses'] = csv

	return out


def plot_residuals(
	S,
	figsize = (8,5),
	singlet_marker = '+',
	multiplet_marker = 'x',
	singlet_alpha = 0.5,
	multiplet_alpha = 1,
	):
	from matplotlib import pyplot as ppl
	
	fig = ppl.figure(figsize = figsize)

	for delta, ylabel, ax, mec in (
		('d13C_VPDB', '$δ^{13}C_{VPDB}$ residuals (‰)', ppl.subplot(211), 'g'),
		('d18O_VPDB', '$δ^{18}O_{VPDB}$ residuals (‰)', ppl.subplot(212), 'r'),
		):
		ppl.sca(ax)
		try:
			X, Y = zip(*[(k, r[f'{delta}_residual']) for k,r in enumerate(S['data']) if f'SE_{delta}' in S['samples'][r['Sample']] and S['samples'][r['Sample']]['N'] == 1])
			ppl.plot(X, Y, singlet_marker, mec = 'k', ms = 4, mew = 1, alpha = singlet_alpha)
		except ValueError:
			pass

		try:
			X, Y = zip(*[(k, r[f'{delta}_residual']) for k,r in enumerate(S['data']) if f'SE_{delta}' in S['samples'][r['Sample']] and S['samples'][r['Sample']]['N'] > 1])
			ppl.plot(X, Y, multiplet_marker, mec = 'k', ms = 4, mew = 1, alpha = multiplet_alpha)
		except ValueError:
			pass

		try:
			X, Y = zip(*[(k, r[f'{delta}_residual']) for k,r in enumerate(S['data']) if f'SE_{delta}' not in S['samples'][r['Sample']] and S['samples'][r['Sample']]['N'] == 1])
			ppl.plot(X, Y, singlet_marker, mec = mec, ms = 4, mew = 1, alpha = singlet_alpha)
		except ValueError:
			pass

		try:
			X, Y = zip(*[(k, r[f'{delta}_residual']) for k,r in enumerate(S['data']) if f'SE_{delta}' not in S['samples'][r['Sample']] and S['samples'][r['Sample']]['N'] > 1])
			ppl.plot(X, Y, multiplet_marker, mec = mec, ms = 4, mew = 1, alpha = multiplet_alpha)
		except ValueError:
			pass

		ppl.axhline(0, color = 'k', lw = 0.4)
		ppl.xticks([])
		ppl.ylabel(ylabel)

	return fig

if __name__ == '__main__':

	x1,y1 = deltas_to_ratios(-10, -20)
	x2,y2 = deltas_to_ratios(10, 20)
	print(ratios_to_deltas(np.linspace(x1,x2,4), np.linspace(y1,y2,4)))
	