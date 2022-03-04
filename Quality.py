from __future__ import print_function,division
import sys
import numpy as np 
import pandas as pd
import dill
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class ClassifierQuality:
	"""This class analyses the classifiers as function of probability and covariate"""
	def __init__(self,file_data,variate,covariate,true_class,covariate_limits=None):
		self.covariate  = covariate
		self.variate    = variate
		self.true_class = true_class

		error_message = "file_data must be DataFrame, string, or list of both."
		usecols = [variate,covariate,true_class]

		if isinstance(file_data,list):
			dfs = []
			for data in file_data:	
				if isinstance(data,str):
					tmp = pd.read_csv(data,usecols=usecols)
					dfs.append(tmp)
				elif isinstance(data,pd.DataFrame):
					tmp = data[usecols].copy()
					dfs.append(tmp)
				else:
					sys.exit(error_message)

				del tmp

			self.dfs = dfs

		elif isinstance(file_data,str):
			df = pd.read_csv(file_data,usecols=usecols)
			self.dfs = [df]
			del df

		elif isinstance(file_data,pd.DataFrame):
			df = file_data[usecols].copy()
			self.dfs = [df]
			del df

		else:
			sys.exit(error_message)

		#--------- Trim df according to covariate_limits -------
		if isinstance(covariate_limits,list):
			list_dfs = []
			for df in self.dfs:
				mask_valid = (df[covariate] > covariate_limits[0]) & \
			             	 (df[covariate] < covariate_limits[1])
				list_dfs.append(df[mask_valid])

			self.dfs = list_dfs
		#-------------------------------------------------------

		self.Ndf  = len(self.dfs)

		self.covariate_min =  np.inf
		self.covariate_max = -np.inf
		
		for df in self.dfs:
			cvmin = df[covariate].min()
			cvmax = df[covariate].max()

			pmin = df[variate].min()
			pmax = df[variate].max()

			if cvmin < self.covariate_min:
				self.covariate_min = cvmin

			if cvmax > self.covariate_max:
				self.covariate_max = cvmax

			assert pmin >= 0.0 and pmin < 1.0,"Probability minimum {0:1.2f}".format(pmin)
			assert pmax <= 1.0 and pmax > 0.0,"Probability maximum {0:1.2f}".format(pmax)

	def confusion_matrix(self,bins=5,prob_steps=100,metric="ACC",
						contamination_rate=None,min_prob=0.01):
		''' Compute the confusion matrix on a grid of prob_steps for each bin of the covariate'''

		#------- Split data frame into bins ---------------------
		if isinstance(bins,int):
			edges = np.linspace(self.covariate_min,self.covariate_max,num=bins,endpoint=False)
		elif isinstance(bins,list):
			edges = np.array(bins)
		else:
			sys.exit("Bins must be integer or list!")

		nbins = len(edges) + 1
		#-------------------------------------------------------

		#-------- Probability thresholds --------------------
		if isinstance(prob_steps,int):
			thresholds =  np.linspace(0,1.0,num=prob_steps,
											endpoint=True)
		elif isinstance(prob_steps,list):
			thresholds = np.array(prob_steps)

		elif isinstance(prob_steps,np.ndarray):
			thresholds =  prob_steps

		elif isinstance(prob_steps,dict):
			to_concat = []
			prev = min_prob
			for k,v in prob_steps.items():
				to_concat.append(np.linspace(
							start=prev,stop=k,num=v,
							endpoint=False))
				prev = k
			thresholds = np.concatenate(to_concat)

		else:
			sys.exit("prob_steps must be list or integer!")

		ns_pro = len(thresholds)
		#----------------------------------------------------

		TP = np.empty((self.Ndf,nbins,ns_pro))
		TN = np.empty((self.Ndf,nbins,ns_pro))
		FP = np.empty((self.Ndf,nbins,ns_pro))
		FN = np.empty((self.Ndf,nbins,ns_pro))
		NS = np.empty((self.Ndf,nbins,ns_pro))

		#------------------------- Loop over dataframes -------------------------------------
		print("Loop over data frames ...")
		for j,df in enumerate(self.dfs):
			print("DF {0}".format(j))
			#------------------- Digitize dataframe --------------------
			bin_cov = np.digitize(df[self.covariate].values,bins=edges)
			#-----------------------------------------------------------

			all_true = df[self.true_class].to_numpy(dtype=bool)
			all_prob = df[self.variate].to_numpy()

			#------------------ Loop over bins --------------------------------------------
			# print("Loop over bins ...")
			for i in range(nbins):
				# print("Bin {0}".format(i))
				if i == 0: # There are no objects in bin zero, so we use it for all objects
					idx = np.arange(len(df))
				else:
					idx = np.where(bin_cov == i)[0]

				#------Select bin sources -----
				trues = all_true[idx]
				probs = all_prob[idx]
				#------------------------------

				#-------- Verify true classes in bin ----------------------
				if i == 0:
					bounds = "[{0:2.1f},{1:2.1f}]".format(self.covariate_min,self.covariate_max)
				elif i == max(bin_cov):
					bounds = "[{0:2.1f},{1:2.1f}]".format(edges[i-1],self.covariate_max)
				else:
					bounds = "[{0:2.1f},{1:2.1f}]".format(edges[i-1],edges[i])

				msg = "class in bin {0} of DataFrame {1}".format(bounds,j)

				assert np.sum( trues) >= 1, "No True  " + msg 
				assert np.sum(~trues) >= 1, "No False " + msg
				#----------------------------------------------------------

				#---------------------------------------------------
				for p,th in enumerate(thresholds):
					TP[j,i,p] = np.logical_and(probs>=th, trues).sum()
					TN[j,i,p] = np.logical_and(probs< th,~trues).sum()
					FP[j,i,p] = np.logical_and(probs>=th,~trues).sum()
					FN[j,i,p] = np.logical_and(probs< th, trues).sum()
					NS[j,i,p] = len(idx)

		print("Computing metrics ...")
		#---------- Metrics ------------------------------------------------------------------------
		CR  = 100.* FP/(FP+TP)
		FPR = 100.* FP/(FP+TN)
		TPR = 100.* TP/(TP+FN)
		PPV = 100.* TP/(TP+FP)
		ACC = 100.* (TP+TN)/(TP+TN+FP+FN)
		MCC = 100.* (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
		F1M = 100.* TP/(TP + 0.5*(FP + FN))
		dCOT = -1.0*np.sqrt((CR -  0.0)**2 + (TPR-100.0)**2)
		dROC = -1.0*np.sqrt((FPR-  0.0)**2 + (TPR-100.0)**2)
		dPRC = -1.0*np.sqrt((PPV-100.0)**2 + (TPR-100.0)**2)
		#-------------------------------------------------------------------------------------------

		#--------------- Creates MultiIndex dataframe -------------------------------------
		iterables = [np.arange(nbins),np.arange(ns_pro)]
		index = pd.MultiIndex.from_product(iterables, names=["bin","level"])
		quality = pd.DataFrame(np.zeros(ns_pro*nbins),columns=["pro"],index=index)
		#----------------------------------------------------------------------------------

		#------------ DF ----------------------------
		quality["pro"]     = np.tile(thresholds,nbins)
		quality["TP"]      = np.mean(TP,axis=0).flatten()
		quality["TN"]      = np.mean(TN,axis=0).flatten()
		quality["FP"]      = np.mean(FP,axis=0).flatten()
		quality["FN"]      = np.mean(FN,axis=0).flatten()
		quality["CR"]      = np.mean(CR,axis=0).flatten()
		quality["FPR"]     = np.mean(FPR,axis=0).flatten()
		quality["TPR"]     = np.mean(TPR,axis=0).flatten()
		quality["PPV"]     = np.mean(PPV,axis=0).flatten()
		quality["ACC"]     = np.mean(ACC,axis=0).flatten()
		quality["MCC"]     = np.mean(MCC,axis=0).flatten()
		quality["F1M"]     = np.mean(F1M,axis=0).flatten()
		quality["dCOT"]    = np.mean(dCOT,axis=0).flatten()
		quality["dROC"]    = np.mean(dROC,axis=0).flatten()
		quality["dPRC"]    = np.mean(dPRC,axis=0).flatten()
		quality["sd_TN"]   = np.std(TN,axis=0).flatten()
		quality["sd_FP"]   = np.std(FP,axis=0).flatten()
		quality["sd_FN"]   = np.std(FN,axis=0).flatten()
		quality["sd_CR"]   = np.std(CR,axis=0).flatten()
		quality["sd_FPR"]  = np.std(FPR,axis=0).flatten()
		quality["sd_TPR"]  = np.std(TPR,axis=0).flatten()
		quality["sd_PPV"]  = np.std(PPV,axis=0).flatten()
		quality["sd_ACC"]  = np.std(ACC,axis=0).flatten()
		quality["sd_MCC"]  = np.std(MCC,axis=0).flatten()
		quality["sd_F1M"]  = np.std(F1M,axis=0).flatten()
		quality["sd_dCOT"] = np.std(dCOT,axis=0).flatten()
		quality["sd_dROC"] = np.std(dROC,axis=0).flatten()
		quality["sd_dPRC"] = np.std(dPRC,axis=0).flatten()
		quality["n_sources"] = np.mean(NS,axis=0).flatten()
		#----------------------------------------------

		#------------------ Loop over bins --------------------------------------------
		# print("Finding optima per bin ...")
		optima = []
		central = []
		for i in range(nbins):
			# print("Bin {0}".format(i))
			#---------------- Identify optimum ------------------------------
			if contamination_rate is None:
				#------------ Metric ------------------------------------------
				try:
					idx_opt = np.nanargmax(quality.loc[(i),metric].to_numpy())
				except ValueError as e:
					print("Error in bin {0}".format(i))
					print(quality.loc[(i)])
					raise e
				#---------------------------------------------------------------
			else:
				#------------ Contamination rate ------------------------------
				try:
					idx_opt = np.nanargmin(
						  np.abs(quality.loc[(i),"CR"].to_numpy() - 
						  contamination_rate))
				except ValueError as e:
					print("Error in bin {0}".format(i))
					print(quality.loc[(i)])
					raise e
				#-------------------------------------------------------------
			#------------------------------------------------------------------

			#------------ Extract ---------------------------------------------
			optimum = quality.loc[[(i,idx_opt)]]
			#----------------------------------------------------------------

			#----------------- Insert values into DataFrames ------------
			if i == 0:
				#------------------ All Covariate values ----------------
				optimum.insert(loc=0,column=self.covariate,value=np.nan)
				optimum.insert(loc=0,column="Strategy",value="All")
				#---------------------------------------------------------
			else:
				#------------ Bining strategy ------------------------------------------
				if i < len(edges):
					value = edges[i-1] + 0.5*(edges[i]-edges[i-1])
				else:
					value = edges[i-1] + 0.5*(self.covariate_max - edges[i-1])
				value = round(value,2)
				central.append(value)
				optimum.insert(loc=0,column=self.covariate,value=value)
				optimum.insert(loc=0,column="Strategy",value="Bin {0}".format(i))
				#-----------------------------------------------------------------------
			#----------------------------------------------------------------------------

			#------- Append ----------------
			optima.append(optimum)
			#-------------------------------

		self.quality = quality
		self.optima  = optima
		self.edges   = edges
		self.central = central

	def plots(self,file_plot,figsize=(10,10),dpi=200,log_scale=False):
		"""Quality Plots """

		#============ Plots ==================================================
		pdf = PdfPages(file_plot)

		#------------------ Color bar and normalization ----------------
		cmap = plt.get_cmap("jet")
		norm = Normalize(vmin=self.covariate_min,vmax=self.covariate_max)
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
		minor_ticks = np.delete(self.edges.copy(),np.argmin(self.edges))
		#----------------------------------------------------------------

		#======================= TPR and CR ================================
		#-------------------- Colorbar --------------------------------------
		fig, axs = plt.subplots(figsize=figsize)
		divider = make_axes_locatable(axs)
		# Add an axes above the main axes.
		cax = divider.append_axes("top", size="7%", pad="2%")
		cbar = fig.colorbar(sm, cax=cax, 
							orientation="horizontal",
							ticklocation="top",
							label=self.covariate,
							ticks=self.central,
							format='%.1f'
							)
		cbar.ax.xaxis.set_ticks(minor_ticks, minor=True)		
		#-----------------------------------------------------------------------------

		
		for i,OP in enumerate(self.optima):
			MU = self.quality.loc[(i)]
			if i==0:
				color    ="black"
				label_tp = "TPR"
				label_cr = "CR"
			else:
				color    = cmap(norm(OP[self.covariate].values[0]))
				label_tp = "_nolegend_"
				label_cr = "_nolegend_"

			axs.fill_between(MU["pro"],MU["TPR"]-MU["sd_TPR"],MU["TPR"]+MU["sd_TPR"],
							color=color,
							zorder=-1,alpha=0.2)
			axs.fill_between(MU["pro"],MU["CR"]-MU["sd_CR"],MU["CR"]+MU["sd_CR"],
							color=color,
							zorder=-1,alpha=0.2)

			axs.plot(MU["pro"],MU["TPR"],c=color,label=label_tp)
			axs.plot(MU["pro"],MU["CR"], c=color,ls="--",label=label_cr)		
			axs.scatter(OP["pro"],OP["TPR"],color=color,marker="X",label="_nolegend_",zorder=10)
			axs.scatter(OP["pro"],OP["CR"],color=color,marker="X",label="_nolegend_",zorder=10)

		
		if log_scale:
			# axs.set_xscale("log")
			axs.set_yscale("log")
		else:
			axs.set_xlim(0,1.0)
			axs.set_ylim(0,100.0)
			axs.set_xticks(np.arange(0,1,step=0.10))
			axs.yaxis.set_major_locator(MultipleLocator(10))
			axs.yaxis.set_major_formatter(FormatStrFormatter('%d'))
			# For the minor ticks, use no labels; default NullFormatter.
			axs.yaxis.set_minor_locator(MultipleLocator(5))

		axs.set_ylabel("Quality indicator [%]")
		axs.set_xlabel("Probability")
		axs.legend(loc="best",ncol=1)
		pdf.savefig(bbox_inches="tight",dpi=dpi)
		plt.close()
		#=========================================================================================

		#======================= TPR ================================
		#-------------------- Colorbar --------------------------------------
		fig, axs = plt.subplots(figsize=figsize)
		divider = make_axes_locatable(axs)
		# Add an axes above the main axes.
		cax = divider.append_axes("top", size="7%", pad="2%")
		cbar = fig.colorbar(sm, cax=cax, 
							orientation="horizontal",
							ticklocation="top",
							label=self.covariate,
							ticks=self.central,
							format='%.1f'
							)
		cbar.ax.xaxis.set_ticks(minor_ticks, minor=True)		
		#-----------------------------------------------------------------------------

		
		for i,OP in enumerate(self.optima):
			MU = self.quality.loc[(i)]
			if i==0:
				color    ="black"
			else:
				color    = cmap(norm(OP[self.covariate].values[0]))

			axs.fill_between(MU["pro"],MU["TPR"]-MU["sd_TPR"],MU["TPR"]+MU["sd_TPR"],
							color=color,
							zorder=-1,alpha=0.2)

			axs.plot(MU["pro"],MU["TPR"],c=color)		
			axs.scatter(OP["pro"],OP["TPR"],color=color,marker="X",zorder=10)

		if log_scale:
			axs.set_yscale("log")
		else:
			axs.set_xlim(0,1.0)
			axs.set_ylim(0,100.0)
			axs.set_xticks(np.arange(0,1,step=0.10))
			axs.yaxis.set_major_locator(MultipleLocator(10))
			axs.yaxis.set_major_formatter(FormatStrFormatter('%d'))
			# For the minor ticks, use no labels; default NullFormatter.
			axs.yaxis.set_minor_locator(MultipleLocator(5))

		axs.set_ylabel("True Positive Rate [%]")
		axs.set_xlabel("Probability")
		pdf.savefig(bbox_inches="tight",dpi=dpi)
		plt.close()
		#=========================================================================================

		#======================= CR ================================
		#-------------------- Colorbar --------------------------------------
		fig, axs = plt.subplots(figsize=figsize)
		divider = make_axes_locatable(axs)
		# Add an axes above the main axes.
		cax = divider.append_axes("top", size="7%", pad="2%")
		cbar = fig.colorbar(sm, cax=cax, 
							orientation="horizontal",
							ticklocation="top",
							label=self.covariate,
							ticks=self.central,
							format='%.1f'
							)
		cbar.ax.xaxis.set_ticks(minor_ticks, minor=True)		
		#-----------------------------------------------------------------------------

		
		for i,OP in enumerate(self.optima):
			MU = self.quality.loc[(i)]
			if i==0:
				color    ="black"
			else:
				color    = cmap(norm(OP[self.covariate].values[0]))

			axs.fill_between(MU["pro"],MU["CR"]-MU["sd_CR"],MU["CR"]+MU["sd_CR"],
							color=color,
							zorder=-1,alpha=0.2)

			axs.plot(MU["pro"],MU["CR"], c=color,ls="--")		
			axs.scatter(OP["pro"],OP["CR"],color=color,marker="X",zorder=10)

		
		if log_scale:
			# axs.set_xscale("log")
			axs.set_yscale("log")
		else:
			axs.set_xlim(0,1.0)
			axs.set_ylim(0,100.0)
			axs.set_xticks(np.arange(0,1,step=0.10))
			axs.yaxis.set_major_locator(MultipleLocator(10))
			axs.yaxis.set_major_formatter(FormatStrFormatter('%d'))
			# For the minor ticks, use no labels; default NullFormatter.
			axs.yaxis.set_minor_locator(MultipleLocator(5))

		axs.set_ylabel("Contamination Rate [%]")
		axs.set_xlabel("Probability")
		pdf.savefig(bbox_inches="tight",dpi=dpi)
		plt.close()
		#=========================================================================================

		#=========================== ROC & PRC ===================================================
		fig, axs = plt.subplots(nrows=1, ncols=2,figsize=figsize)
		# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.01)
		#-------------------- Colorbar --------------------------------------
		cbar = fig.colorbar(sm, 
							ax = axs,
							orientation="horizontal",
							ticklocation="top",
							label=self.covariate,
							ticks=self.central,
							format='%.1f',
							pad=0.07
							)
		cbar.ax.xaxis.set_ticks(minor_ticks, minor=True)		
		#-----------------------------------------------------------------------------

		for ax,m,title in zip(axs.flatten(),[["FPR","TPR"],["TPR","PPV"]],["ROC","PRC"]):
			for i,OP in enumerate(self.optima):
				MU = self.quality.loc[(i)]
				if i==0:
					color   ="black"
				else:
					color   = cmap(norm(OP[self.covariate].values[0]))

				ax.fill_between(MU[m[0]],MU[m[1]]-MU["sd_"+m[1]],MU[m[1]]+MU["sd_"+m[1]],
								color=color,
								zorder=0,alpha=0.2)
				ax.plot(MU[m[0]],MU[m[1]],c=color,zorder=1)	
				ax.scatter(OP[m[0]],OP[m[1]],color=color,marker="X",zorder=2)

			ax.set_xlabel(m[0],labelpad=0)
			ax.set_ylabel(m[1],labelpad=0)
			if log_scale:
				ax.set_xscale("log")
				ax.set_yscale("log")

			ax.set_title(title)
		pdf.savefig(bbox_inches="tight",dpi=dpi)
		plt.close()
		#=========================================================================================

		#========================= Metrics =======================================================
		fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=figsize)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.0)
		#-------------------- Colorbar --------------------------------------
		cbar = fig.colorbar(sm, 
							ax = axs,
							orientation="horizontal",
							ticklocation="top",
							label=self.covariate,
							ticks=self.central,
							format='%.1f',
							pad=0.07
							)
		cbar.ax.xaxis.set_ticks(minor_ticks, minor=True)		
		#-----------------------------------------------------------------------------

		for ax,m in zip(axs.flatten(),["F1M","ACC","MCC"]):
			for i,OP in enumerate(self.optima):
				MU = self.quality.loc[(i)]
				if i==0:
					color   ="black"
				else:
					color   = cmap(norm(OP[self.covariate].values[0]))

				ax.fill_between(MU["pro"],MU[m]-MU["sd_"+m],MU[m]+MU["sd_"+m],
								color=color,
								zorder=0,alpha=0.2)
				ax.plot(MU["pro"],MU[m],c=color,zorder=1)	
				ax.scatter(OP["pro"],OP[m],color=color,marker="X",zorder=2)

			if log_scale:
				# ax.set_xscale("log")
				ax.set_yscale("log")
			ax.set_ylabel(m,labelpad=0)
		axs[2].set_xlabel("Probability")
		pdf.savefig(bbox_inches="tight",dpi=dpi)
		plt.close()


		fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True,figsize=figsize)
		plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.0)
		#-------------------- Colorbar --------------------------------------
		cbar = fig.colorbar(sm, 
							ax = axs,
							orientation="horizontal",
							ticklocation="top",
							label=self.covariate,
							ticks=self.central,
							format='%.1f',
							pad=0.07
							)
		cbar.ax.xaxis.set_ticks(minor_ticks, minor=True)		
		#-----------------------------------------------------------------------------

		for ax,m in zip(axs.flatten(),["dCOT","dROC","dPRC"]):
			for i,OP in enumerate(self.optima):
				MU = self.quality.loc[(i)]
				if i==0:
					color   ="black"
				else:
					color   = cmap(norm(OP[self.covariate].values[0]))

				ax.fill_between(MU["pro"],MU[m]-MU["sd_"+m],MU[m]+MU["sd_"+m],
								color=color,
								zorder=0,alpha=0.2)
				ax.plot(MU["pro"],MU[m],c=color,zorder=1)	
				ax.scatter(OP["pro"],OP[m],color=color,marker="X",zorder=2)

			ax.set_ylabel(m,labelpad=0)
		axs[2].set_xlabel("Probability")
		pdf.savefig(bbox_inches="tight",dpi=dpi)
		plt.close()

		#========================================================================================

		pdf.close()

	def save(self,file_tex):
		tab = pd.concat(self.optima,ignore_index=True,sort=False)
		tab.set_index("Strategy",inplace=True)

		#-------------- Save as latex ---------------------------------
		tab = tab.loc[:,[self.covariate,"pro","n_sources",
						"TP","FP","TN","FN",
						"TPR","CR","FPR",
						self.metric]]
		tab.to_latex(file_tex,column_format=11*"|c" + "|",
						float_format="%.2f",na_rep="-",escape=False)
		#--------------------------------------------------------------


		#-------------- pickle ---------------------
		quality = {"edges":self.edges,
				   "thresholds":tab["pro"],
				   "covariate":self.covariate}
		with open(file_tex.replace(".tex",".pkl"), 'wb') as out_strm: 
			dill.dump(quality, out_strm,protocol=2)