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
	def __init__(self,file_data,variate,covariate,true_class):
		self.covariate  = covariate
		self.variate    = variate
		self.true_class = true_class

		error_message = "file_data must be DataFrame, string, or list of both."
		usecols = [variate,covariate,true_class]

		if isinstance(file_data,list):
			dfs = []
			for data in file_data:	
				if isinstance(data,str):
					df = pd.read_csv(data,usecols=usecols)
					dfs.append(df)
				elif isinstance(data,pd.DataFrame):
					df = data[usecols].copy()
					dfs.append(df)
				else:
					sys.exit(error_message)

			self.df  = pd.concat(dfs,ignore_index=True)
			self.dfs = dfs

		elif isinstance(file_data,str):
			self.df = pd.read_csv(file_data,usecols=usecols)
			self.dfs = [self.df]

		elif isinstance(file_data,pd.DataFrame):
			self.df = file_data[usecols].copy()
			self.dfs = [self.df]

		else:
			sys.exit(error_message)

		self.vmin = self.df[covariate].min()
		self.vmax = self.df[covariate].max()


	def confusion_matrix(self,bins=5,prob_steps=100,metric="ACC",contamination_rate=None):
		''' Compute the confusion matrix on a grid of prob_steps for each bin of the covariate'''

		#------- Split data frame into bins ---------------------
		if isinstance(bins,int):
			edges = np.linspace(self.vmin,self.vmax,num=bins,endpoint=False)
		elif isinstance(bins,list):
			edges = np.array(bins)
		else:
			sys.exit("Bins must be integer or list!")
		#-------------------------------------------------------

		#--------------- Creates MultiIndex dataframe -------------------------------------
		iterables = [np.arange(len(self.dfs)),np.arange(len(edges)+1),np.arange(prob_steps)]
		index = pd.MultiIndex.from_product(iterables, names=["case","bin","level"])
		CM = pd.DataFrame(np.zeros(prob_steps*(len(edges)+1)*len(self.dfs)),
							columns=["pro"],index=index)
		#----------------------------------------------------------------------------------

		thresholds =  np.linspace(0,0.999,num=prob_steps,endpoint=True)

		#------------------------- Loop over dataframes -------------------------------------
		for j,df in enumerate(self.dfs):
			#------------------- Digitize dataframe --------------------
			bin_cov = np.digitize(df[self.covariate].values,bins=edges)
			#-----------------------------------------------------------

			#------------------ Loop over bins --------------------------------------------
			Ns = []
			for i in range(max(bin_cov)+1):
				if i == 0: # There are no objects in bin zero, so we use it for all objects
					idx = np.arange(len(df))
				else:
					idx = np.where(bin_cov == i)[0]

				#-- Temporary dataframe --
				tmp = df.iloc[idx].copy()
				#-------------------------

				#-----------------------------------------------------
				trues = tmp[self.true_class].to_numpy()
				probs = tmp[self.variate].to_numpy()
				TP = np.empty(prob_steps)
				TN = np.empty(prob_steps)
				FP = np.empty(prob_steps)
				FN = np.empty(prob_steps)
				for p,th in enumerate(thresholds):
					TP[p] = np.logical_and(probs>=th, trues).sum()
					TN[p] = np.logical_and(probs< th,~trues).sum()
					FP[p] = np.logical_and(probs>=th,~trues).sum()
					FN[p] = np.logical_and(probs< th, trues).sum()


				#----- Insert values ------------
				CM.loc[(j,i),"pro"] = thresholds
				CM.loc[(j,i),"TP"]  = TP
				CM.loc[(j,i),"TN"]  = TN
				CM.loc[(j,i),"FP"]  = FP
				CM.loc[(j,i),"FN"]  = FN
				#-------------------------------

				#---- n_sources --------------------
				CM.loc[(j,i),"n_sources"] = len(idx)
				#-----------------------------------

		#---------- Metrics ------------------------------------------------------------------------
		CM["CR"]  = 100.* CM["FP"] / (CM["FP"]+CM["TP"])
		CM["FPR"] = 100.* CM["FP"] / (CM["FP"]+CM["TN"])
		CM["TPR"] = 100.* CM["TP"] / (CM["TP"]+CM["FN"])
		CM["PPV"] = 100.* CM["TP"] / (CM["TP"]+CM["FP"])
		CM["ACC"] = 100.* (CM["TP"] + CM["TN"]) / (CM["TP"]+CM["TN"]+CM["FP"]+CM["FN"])
		CM["MCC"] = 100.* (CM["TP"]*CM["TN"] - CM["FP"]*CM["FN"])/\
			np.sqrt((CM["TP"]+CM["FP"])*(CM["TP"]+CM["FN"])*(CM["TN"]+CM["FP"])*(CM["TN"]+CM["FN"]))
		CM["F1M"] = 100.* CM["TP"] / (CM["TP"] + 0.5*(CM["FP"] + CM["FN"]))
		CM["dCOT"] = -1.0*np.sqrt((CM["CR"]-0.0)**2 + (CM["TPR"]-100.0)**2)
		CM["dROC"] = -1.0*np.sqrt((CM["FPR"]-0.0)**2 + (CM["TPR"]-100.0)**2)
		CM["dPRC"] = -1.0*np.sqrt((CM["PPV"]-100.0)**2 + (CM["TPR"]-100.0)**2)
		#-------------------------------------------------------------------------------------------

		#------------ Groupby------------------
		grp = CM.groupby(level=["bin","level"])
		quality_mean = grp.mean()
		quality_std  = grp.std()
		#--------------------------------------

		#------------------ Loop over bins --------------------------------------------
		optima = []
		central = []
		for i in range(len(edges)+1):
			#---------------- Identify optimum ------------------------------
			if contamination_rate is None:
				#------------ Metric ------------------------------------------
				idx_opt = np.nanargmax(quality_mean.loc[(i),metric].to_numpy())
				#---------------------------------------------------------------
			else:
				#------------ Contamination rate ------------------------------
				idx_opt = np.nanargmin(
						  np.abs(quality_mean.loc[(i),"CR"].to_numpy() - 
						  contamination_rate))
				#-------------------------------------------------------------
			#------------------------------------------------------------------

			#------------ Extract ---------------------------------------------
			optimum_mu = quality_mean.loc[[(i,idx_opt)]]
			optimum_sd = quality_mean.loc[[(i,idx_opt)]]
			optimum    = optimum_mu.merge(optimum_sd,suffixes=("","_sd"))
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
					value = edges[i-1] + 0.5*(self.vmax - edges[i-1])
				value = round(value,2)
				central.append(value)
				optimum.insert(loc=0,column=self.covariate,value=value)
				optimum.insert(loc=0,column="Strategy",value="Bin {0}".format(i))
				#-----------------------------------------------------------------------
			#----------------------------------------------------------------------------

			#------- Append ----------------
			optima.append(optimum)
			#-------------------------------

		self.quality_mu = quality_mean
		self.quality_sd = quality_std
		self.optima  = optima
		self.edges   = edges
		self.central = central

	def plots(self,file_plot,figsize=(10,10),dpi=200):
		"""Quality Plots """

		#============ Plots ==================================================
		pdf = PdfPages(file_plot)

		#------------------ Color bar and normalization ----------------
		cmap = plt.get_cmap("jet")
		norm = Normalize(vmin=self.vmin,vmax=self.vmax)
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
			MU = self.quality_mu.loc[(i)]
			SD = self.quality_sd.loc[(i)]
			if i==0:
				color    ="black"
				label_tp = "TPR"
				label_cr = "CR"
			else:
				color    = cmap(norm(OP[self.covariate].values[0]))
				label_tp = "_nolegend_"
				label_cr = "_nolegend_"

			axs.fill_between(MU["pro"],MU["TPR"]-SD["TPR"],MU["TPR"]+SD["TPR"],
							color=color,
							zorder=-1,alpha=0.2)
			axs.fill_between(MU["pro"],MU["CR"]-SD["CR"],MU["CR"]+SD["CR"],
							color=color,
							zorder=-1,alpha=0.2)

			axs.plot(MU["pro"],MU["TPR"],c=color,label=label_tp)
			axs.plot(MU["pro"],MU["CR"], c=color,ls="--",label=label_cr)		
			axs.scatter(OP["pro"],OP["TPR"],color=color,marker="X",label="_nolegend_",zorder=10)
			axs.scatter(OP["pro"],OP["CR"],color=color,marker="X",label="_nolegend_",zorder=10)

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

		#=========================== ROC & PRC ===================================================
		fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True,figsize=figsize)
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

		for ax,m,title in zip(axs.flatten(),[["FPR","TPR"],["TPR","PPV"]],["ROC","PRC"]):
			for i,OP in enumerate(self.optima):
				MU = self.quality_mu.loc[(i)]
				SD = self.quality_sd.loc[(i)]
				if i==0:
					color   ="black"
				else:
					color   = cmap(norm(OP[self.covariate].values[0]))

				ax.fill_between(MU[m[0]],MU[m[1]]-SD[m[1]],MU[m[1]]+SD[m[1]],
								color=color,
								zorder=0,alpha=0.2)
				ax.plot(MU[m[0]],MU[m[1]],c=color,zorder=1)	
				ax.scatter(OP[m[0]],OP[m[1]],color=color,marker="X",zorder=2)

			ax.set_xlabel(m[0],labelpad=0)
			ax.set_ylabel(m[1],labelpad=0)
			ax.set_xlim(0,100.0)
			ax.set_ylim(0,100.0)
			ax.set_aspect("equal")
			ax.set_title(title)
		pdf.savefig(bbox_inches="tight",dpi=dpi)
		plt.close()
		#=========================================================================================

		#========================= Metrics =======================================================
		fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True,figsize=figsize)
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

		for ax,m in zip(axs.flatten(),["F1M","ACC","MCC","dCOT","dROC","dPRC"]):
			for i,OP in enumerate(self.optima):
				MU = self.quality_mu.loc[(i)]
				SD = self.quality_sd.loc[(i)]
				if i==0:
					color   ="black"
				else:
					color   = cmap(norm(OP[self.covariate].values[0]))

				ax.fill_between(MU["pro"],MU[m]-SD[m],MU[m]+SD[m],
								color=color,
								zorder=0,alpha=0.2)
				ax.plot(MU["pro"],MU[m],c=color,zorder=1)	
				ax.scatter(OP["pro"],OP[m],color=color,marker="X",zorder=2)

			ax.set_xlim(0,1.0)
			ax.set_ylabel(m,labelpad=0)
		axs[2,0].set_xlabel("Probability")
		axs[2,1].set_xlabel("Probability")
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
						"TPR","CR","FPR","PPV","ACC","F1M","MCC","dCOT","dROC","dPRC"]]
		tab.to_latex(file_tex,column_format=17*"|c" + "|",
						float_format="%.2f",na_rep="-",escape=False)
		#--------------------------------------------------------------


		#-------------- pickle ---------------------
		quality = {"edges":self.edges,
				   "thresholds":tab["pro"],
				   "covariate":self.covariate}
		with open(file_tex.replace(".tex",".pkl"), 'wb') as out_strm: 
			dill.dump(quality, out_strm,protocol=2)