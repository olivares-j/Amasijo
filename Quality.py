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
		self.df = pd.read_csv(file_data,usecols=[variate,covariate,true_class])


	def confusion_matrix(self,bins=5,prob_steps=100):
		''' Compute the confusion matrix on a grid of prob_steps for each bin of the covariate'''

		#------- Split data frame into bins ---------------------
		if isinstance(bins,int):
			edges = np.linspace(self.df[self.covariate].min(),self.df[self.covariate].max(),
								num=bins,endpoint=False)
		elif isinstance(bins,list):
			edges = np.array(bins)
		else:
			sys.exit("Bins must be integer or list!")
		bin_cov = np.digitize(self.df[self.covariate].values,bins=edges)
		#-------------------------------------------------------

		#----------- Loop over bins ----------------------------
		CMs = []
		optima = []

		for i in range(max(bin_cov)+1):
			if i == 0: # There are no objects in bin zero, so we use it for all objects
				idx = np.arange(len(self.df))
			else:
				idx = np.where(bin_cov == i)[0]

			CM = pd.DataFrame(np.linspace(0,1,num=prob_steps,endpoint=True),columns=["pro"])

			#---------- Positives and Negatives -------------------------------------------------
			CM["TP"] = CM["pro"].apply(lambda pro: np.sum(
						(self.df[self.variate].iloc[idx] >= pro) &
						self.df[self.true_class].iloc[idx]
						))
			CM["TN"] = CM["pro"].apply(lambda pro: np.sum(
						(self.df[self.variate].iloc[idx] < pro)	&   
						~self.df[self.true_class].iloc[idx]
						))
			CM["FP"] = CM["pro"].apply(lambda pro: np.sum(
						(self.df[self.variate].iloc[idx] >= pro) &  
						~self.df[self.true_class].iloc[idx]
						))
			CM["FN"] = CM["pro"].apply(lambda pro: np.sum(
						(self.df[self.variate].iloc[idx] < pro)  &    
						self.df[self.true_class].iloc[idx]
						))

			#---------- Quality indicators ----------------------------------------------------
			CM["CR"]  = 100.* CM["FP"] / (CM["FP"]+CM["TP"])
			CM["FPR"] = 100.* CM["FP"] / (CM["FP"]+CM["TN"])
			CM["TPR"] = 100.* CM["TP"] / (CM["TP"]+CM["FN"])
			CM["PPV"] = 100.* CM["TP"] / (CM["TP"]+CM["FP"])
			CM["ACC"] = 100.* (CM["TP"] + CM["TN"]) / (CM["TP"]+CM["TN"]+CM["FP"]+CM["FN"])

			CM["MCC"] = 100. * (CM["TP"]*CM["TN"] - CM["FP"]*CM["FN"])/np.sqrt(
				(CM["TP"]+CM["FP"])*(CM["TP"]+CM["FN"])*(CM["TN"]+CM["FP"])*(CM["TN"]+CM["FN"]))

			CM["DST"] = np.sqrt((CM["CR"]-0.0)**2 + (CM["TPR"]-1.0)**2)

			#---------------- Identify optimum ----------
			idx_opt = np.nanargmax(CM["ACC"].to_numpy())
			optimum = CM.loc[[idx_opt]]
			opt_pro = CM["pro"].iloc[[idx_opt]].values[0]
			#--------------------------------------------

			#----------------- Insert values into DataFrames -------------------------
			optimum.insert(loc=0,column="n_sources",value=len(idx))
			if i == 0:
				#------------------ All Covariate values -----------------------
				optimum.insert(loc=0,column=self.covariate,value=np.nan)
				optimum.insert(loc=0,column="Strategy",value="All")
				CM.insert(loc=0,column="Strategy",value="All")
				#---------------------------------------------------------
			else:
				#------------ Bining strategy ------------------------------------------
				optimum.insert(loc=0,column=self.covariate,
								value=np.mean(self.df[self.covariate].iloc[idx]))
				optimum.insert(loc=0,column="Strategy",value="Bin {0}".format(i))
				CM.insert(loc=0,column="Strategy",value="Bin {0}".format(i))
				#-----------------------------------------------------------------------
			#----------------------------------------------------------------------------

			#------- Append ----------------
			CMs.append(CM)
			optima.append(optimum)
			#-------------------------------

		self.CMs = CMs
		self.optima = optima
		self.edges = edges

	def plots(self,file_plot,figsize=(6,6),dpi=200):
		"""Quality Plots """

		#============ Plots ==================================================
		pdf = PdfPages(file_plot)

		cmap = plt.get_cmap("jet")
		norm = Normalize(vmin=self.df[self.covariate].min(),
						 vmax=self.df[self.covariate].max())
		sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])

		#-------------------- Colorbar --------------------------------------
		ymin,ymax = 99.0,100.0
		minor_ticks = self.edges
		major_ticks = self.edges + 0.5*(self.edges[1]-self.edges[0])
		fig, axs = plt.subplots(figsize=figsize)

		divider = make_axes_locatable(axs)
		# Add an axes above the main axes.
		cax = divider.append_axes("top", size="7%", pad="2%")
		cbar = fig.colorbar(sm, cax=cax, 
							orientation="horizontal",
							ticklocation="top",
							label=self.covariate,
							ticks=major_ticks,
							format='%.1f'
							)
		# cax0.xaxis.set_label_position("top")
		# cax0.xaxis.set_ticks_position("top")
		cbar.ax.xaxis.set_ticks(minor_ticks, minor=True)
		# # Change tick position to top (with the default tick position "bottom", ticks
		# # overlap the image).
		# 
		
		# #-----------------------------------------------------------------------------

		#-------------------- TPR and CR ---------------------------------------------------------------
		axs.plot(self.CMs[0]["pro"],self.CMs[0]["TPR"],color="black",         label="TPR")
		axs.plot(self.CMs[0]["pro"],self.CMs[0]["CR"],color="black",ls="--", label="CR")
		axs.scatter(self.optima[0]["pro"],self.optima[0]["TPR"],c="black",marker="X",label="_nolegend_",zorder=10)
		axs.scatter(self.optima[0]["pro"],self.optima[0]["CR"],c="black",marker="X",label="_nolegend_",zorder=10)
		for tmp,opt in zip(self.CMs[1:],self.optima[1:]):
			axs.plot(tmp["pro"],tmp["TPR"],c=cmap(norm(opt[self.covariate].values[0])),label="_nolegend_")
			axs.plot(tmp["pro"],tmp["CR"],c=cmap(norm(opt[self.covariate].values[0])),ls="--",label="_nolegend_")		
			axs.scatter(opt["pro"],opt["TPR"],c=opt[self.covariate].values,cmap=cmap,norm=norm,
												marker="X",label="_nolegend_",zorder=10)
			axs.scatter(opt["pro"],opt["CR"],c=opt[self.covariate].values,cmap=cmap,norm=norm,
												marker="X",label="_nolegend_",zorder=10)

		axs.set_xlim(0,1.0)
		axs.set_ylim(0,100.0)
		axs.set_xticks(np.arange(0,1,step=0.10))
		axs.yaxis.set_major_locator(MultipleLocator(10))
		axs.yaxis.set_major_formatter(FormatStrFormatter('%d'))
		# For the minor ticks, use no labels; default NullFormatter.
		axs.yaxis.set_minor_locator(MultipleLocator(5))
		axs.set_ylabel("Quality indicator [%]")
		axs.set_xlabel("Probability")
		axs.legend(loc="center left",ncol=1)
		pdf.savefig(bbox_inches="tight",dpi=dpi)
		plt.close()
		pdf.close()

	def save(self,file):
		tab = pd.concat(self.optima,ignore_index=True,sort=False)
		tab.set_index("Strategy",inplace=True)

		#-------------- Save as latex ---------------------------------
		tab = tab.loc[:,[self.covariate,"pro","n_sources","TP","FP",
						"TPR","CR","FPR","PPV","ACC","MCC","DST"]]
		tab.to_latex(file.replace(".pkl",".tex"),column_format=13*"|c" + "|",
						float_format="%.2f",na_rep="-",escape=False)
		#--------------------------------------------------------------


		#-------------- pickle ---------------------
		quality = {"edges":self.edges,"thresholds":tab["pro"]}
		with open(file, 'wb') as out_strm: 
			dill.dump(quality, out_strm,protocol=2)