This file is part of the documentation supplementing code and data for the paper
     "Reproducible Air Passenger Demand Estimation"
     (A. M. Tillmann, I. Joormann, S. Ammann)
and provides descriptions of the provided data as well as external data,
including source-file links.

It is generally assumed that downloaded external data from Eurostat follows the
NUTS-2021 classification scheme. However, since the transition from the older
NUTS-2016 codes is yet to be completed, we provide additional data that allows
to combine schemes (where possible), or work solely based on NUTS-2016.
Full details on changes from NUTS-2016 to NUTS-2021 can be found in the file
https://ec.europa.eu/eurostat/documents/345175/629341/NUTS2021.xlsx
available from the Eurostat website; see also point (4) below.
Thus, the extract_data.py scheme can work in legacy mode (using NUTS-2016 codes)
or in transition mode (using NUTS-2021 codes, mapping data that is currently
only available for NUTS-2016 codes to the new classifications where possible).


CONTENTS:
(1) provided (base) data
(2) data files created by extract_data.py
(3) external data files used for (2), downloadable with download_data.py
(4) further remarks



*************************************************************************************
*****			(1)	provided (base) data:				*****
*************************************************************************************

base-data.csv
	This file contains IDs, aggregated ICAO-codes, name/country, NUTS-3 localization,
	metropolitan region code, aggregated NUTS-3 codes and (aggregated) NUTS2-codes
	for our set of 531 pseudo-airports within Europe.

additional_data/dist-data.csv
	This file contains the (kilometer-) distance matrix for all pairs of the
	531 pseudo-airports (in order of IDs assigned in base-data.csv).

additional_data/CH_cantons_to_NUTS3.csv
	This file provides a mapping from the names of Swiss cantons to the equivalent
	NUTS-3 codes (identical in NUTS-2016 and NUTS-2021 classification scheme).

additional_data/UK_NUTS3_to_ITL3.csv
	This file provides a mapping from NUTS-3 codes to the ITL3-codes used by
	the UK's Office of National Statistics (including region names).
	The mapping is based on a name-match between NUTS-3 and ITL3 regions;
	a list of the latter including the respective region names can be found at
	https://geoportal.statistics.gov.uk/documents/international-territorial-levels-level-3-january-2021-names-and-codes-in-the-united-kingdom-v2
	(for level 1 and 2 regions/nomenclature, see also
	https://geoportal.statistics.gov.uk/datasets/ons::international-territorial-levels-level-2-january-2021-names-and-codes-in-the-united-kingdom/explore
	https://geoportal.statistics.gov.uk/datasets/ons::international-territorial-levels-level-1-january-2021-names-and-codes-in-the-united-kingdom/explore)

additional_data/Coastal-regions-NUTS-2021_amended.csv
	This file provides a list indicating for each of the NUTS-3 regions
	that occur in our data set whether the region has an ocean coastline.
	The data was based on the NUTS-2016 file obtainable from Eurostat via
	https://ec.europa.eu/eurostat/documents/1797762/1797951/Coastal-regions-NUTS-2016.xlsx,
	and expanded significantly by manually identifying missing regions from the maps
	that can be found in "Regions in the European Union, edition 2018"
	(https://ec.europa.eu/eurostat/documents/3859598/9397402/KS-GQ-18-007-EN-N.pdf).
	Note: The difference stems from the fact that Eurostat no longer defines coastal
	regions simply as NUTS-3 regions having an ocean coast, as they apparently did until 2006
	and as we do here, but instead, they additionally require that within NUTS-3 regions
	with an ocean coast, more than 50% of the population lives no more than 50km from the coast.
	Moreover, the regions in Coastal-regions-NUTS-2016.xlsx are inconsistent with the description
	given on https://ec.europa.eu/eurostat/de/web/coastal-island-outermost-regions/methodology
	(the numbers do not add up). Therefore, we manually checked and amended the list,
	and incorporated recodings to the NUTS-2021 classification scheme.

additional_data/NUTS2016to2021recodings.csv
	This file provides a mapping from NUTS-3 codes in the NUTS-2021
	classification scheme to their respective correspondences in the
	NUTS-2016 scheme. Since several regions were not simply recoded 
	but also involved splits, merges, and boundary changes, this
	mapping is not exact in terms of actual	NUTS-3 regions, but it
	is exact for the purposes of this work, in the sense that new
	codes of a region containing some (pseudo-)airport are matched
	to the old code(s) of the region containing the same airport(s).
	The file also lists NUTS-2 codes (NUTS-2021 scheme) that do not
	allow for a backward-matching to NUTS-2016 because they were
	created by merging and/or splitting old NUTS-2 codes.
	
legacy_NUTS2016/base-data_NUTS2016.csv
	This is the base-data file using NUTS-2016 codes (534 pseudo-airports).
legacy_NUTS2016/dist-data_NUTS2016.csv
	This is the dist-data file for use with NUTS-2016 base data.
legacy_NUTS2016/UK_NUTS3_to_ITL3_NUTS2016.csv
	This is the UK_NUTS3_to_ITL3 file using NUTS-2016 codes.
	This mapping is based on name-match between NUTS-3 and ITL3 regions,
	where a list of the latter including the respective region names was
	obtained from an earlier version of the above-linked file that is no
	longer available online.
legacy_NUTS2016/Coastal-regions-NUTS-2016_amended.csv
	This is the list of coastal region classifications, as described
	above but using NUTS-2016 codes.
legacy_NUTS2016/nama_10r_3gdp_NUTS2016.tsv
	This is an older version of the "nama_10r_3gdp.tsv" file that
	can be obtained from the Eurostat database (link: see below). 
	Since this file is no longer available from Eurostat, but was
	distributed under the CC BY 4.0 license, we can and do provide
	it here to enable accessing GDP data (up to 2018) by NUTS-2016
	codes; it was downloaded on June 18, 2021.
	(The current version of this file already adheres to the
	NUTS-2021 classification scheme.)


*************************************************************************************
*****		(2)	data files created by extract_data.py:			*****
*************************************************************************************
The filenames given here are default names; they can be changed by setting the
corresponding arguments of extract_data.py, cf. its help (python3 extract_data.py -h).

stat-data.csv
	This file collects the data associated with individual (pseudo-)airports.
	
pairwise-stat-data.csv
	This file collects the data associated with routes, i.e., pairs of (pseudo-)
	airports.
	(Note: The "international" feature is not listed, as it can easily be derived
	from the "domestic" feature as the two are complementary.)

passengers-on-board-data.csv
	This file can be created optionally and then contains the passenger volume 
	matrix for all pairs of (pseudo-)airports in the data set.

extract.log
	This log file documents some data set statistics (if enabled, see help for
	extract_data.py) and issues encountered during extraction code execution.


*************************************************************************************
*****		(3)	external data files (cf. download_data.py):		*****
*************************************************************************************
The following files can be downloaded using download_data.py, or when running
extract_data.py with the --downloadData flag; they are stored in a folder called
"external_data".

prc_ppp_ind.tsv
	Eurostat NUTS-1 price level index (PLI) data,
	source: https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/prc_ppp_ind.tsv.gz

nama_10r_3gdp.tsv
	Eurostat NUTS-3 GDP data for all countries (except UK and CH),
	source: https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/nama_10r_3gdp.tsv.gz
	
ert_bil_eur_a.tsv
	Eurostat exchange rate data GBP and CHF to EUR,
	source: https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/ert_bil_eur_a.tsv.gz
	
demo_r_pjanaggr3.tsv
	Eurostat NUTS-2 and NUTS-3 population data,
	source: https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/demo_r_pjanaggr3.tsv.gz
tour_occ_nin2.tsv
	Eurostat NUTS-2-level data on nights spent in tourist accommodations,
	source: https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tour_occ_nin2.tsv.gz
	
Islands-regions-NUTS-2016.csv
Islands-regions-NUTS-2021.csv
	Eurostat NUTS-3 level indicators for on-island geolocation,
	sources: https://ec.europa.eu/eurostat/documents/1797762/1797951/Island-regions-NUTS-2016.xlsx
		 https://ec.europa.eu/eurostat/documents/1797762/1797951/Island-regions-NUTS-2021.xlsx

	Depending on whether extract_data.py is run in legacy- or transition-mode (flag --legacy or --transition, resp.),
	either the NUTS-2016 or the NUTS-2021 version of this file is needed. If you want to experiment with both modes,
	you will need to temporarily copy the existing Islands...-file to a different folder than external_data, because
	currently, the entire external_data folder needs to be deleted before one can download the data anew (e.g., with
	a different flag, to obtain the respective other Islands...-file); this holds for both download options (via
	download_data.py or extract_data.py with --downloadData flag). Afterwards, the temporarily moved file can safely
	be moved back into the external_data folder (where it must be located for the corresponding extract_data.py run
	to work properly). 

	Note: This list of "island" regions does NOT encompass all regions that simply lie on an island (or consist only of islands)!
	For example, technically, all of the United Kingdom is an island, but only some NUTS-3 regions are listed here; similarly,
	parts of Denmark are actually islands (though connected to the mainland by bridges), but only the island of Bornholm (which
	has its own NUTS-3 code) is listed here as an island. One could amend this list by going through the "Regions in the European
	Union"-booklet, as we did w.r.t. coastal regions, but for simplicity, we did not do this for the island list.
	  
ilc_peps11.csv
	Eurostat NUTS-1-level data on percentages of people at risk of poverty or social exclusion
	source: https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/ilc_peps11.tsv.gz

avia_par_xx.tsv
	Eurostat data for most important flight connections to/from airports in country XX;
	"passengers-on-board" serves as our proxy for air passenger volume/demand.
	source: https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/avia_par_xx.tsv.gz

	These files exist for XX replaced by any and all of the countries listed in the following (note that in the URL
	and file names, corresponding lowercase letters of the two-letter country code must be/are used):
	AT -- Austria          BE -- Belgium          BG -- Bulgaria          CH -- Switzerland
	CY -- Cyprus           CZ -- Czech Republic   DE -- Germany           DK -- Denmark
	EE -- Estland          EL -- Greece           ES -- Spain             FI -- Finland
	FR -- France           HR -- Croatia          HU -- Hungary           IE -- Ireland
	IS -- Iceland          IT -- Italy            LT -- Lithuania         LU -- Luxemburg
	LV -- Latvia           ME -- Montenegro       MK -- North Macedonia   MT -- Malta
        NL -- The Netherlands  NO -- Norway           PL -- Poland            PT -- Portugal
        RO -- Romania          RS -- Serbia           SE -- Sweden            SI -- Slovenia
        SK -- Slovakia         TR -- Turkey           UK -- United Kingdom

	Unfortunately, no such data appears to exist yet for the EU member candidate country of Albania (country code AL),
	though this might change in the future. (Nevertheless, one Albanian airport -- LATI -- is among the important
	connections from other countries, and so is present in the data sets.)

regionalgrossdomesticproductallitlregions.csv
	UK's Office of National Statistics GDP data for UK's ITL3-regions (corresp. to NUTS-3),
	source: https://www.ons.gov.uk/file?uri=%2feconomy%2fgrossdomesticproductgdp%2fdatasets%2fregionalgrossdomesticproductallnutslevelregions%2f1998to2019/regionalgrossdomesticproductallitlregions.xlsx
	(Table 5 of the linked spreadsheet)

je-e-04.02.06.01.csv
	Switzerland's Federal Statistical Office's GDP data for CH's cantons (corresp. NUTS-3),
	source: https://dam-api.bfs.admin.ch/hub/api/dam/assets/19544503/master
	(URL downloads file called je-e-04-02-06-01.xlsx)


*************************************************************************************
*****			(4)	further remarks:				*****
*************************************************************************************
Eurostat defines metropolitan areas (as collections of NUTS-3 regions); for NUTS-2016
classification, a list of these metro regions can be found at
        https://ec.europa.eu/eurostat/documents/4313761/4311719/Metro-regions-NUTS-2016.xlsx
and for NUTS-2021 classification, there is a corresponding table in the spreadsheet
    	https://ec.europa.eu/eurostat/documents/345175/629341/NUTS2021.xlsx
This latter spreadsheet also contains details on the changes from NUTS-2016 to
NUTS-2021 classification, which we used to (manually) update NUTS-2016-based base data
and create the mapping file additional_data/NUTS2016to2021recodings.csv.

Airport localizations in NUTS-3 regions were found out manually with the help of
Eurostat NUTS-3 region maps (see https://ec.europa.eu/eurostat/web/nuts/nuts-maps),
Google Maps (https://www.google.com/maps/), and a combination of web search for an
airport's postal code and the Eurostat/TERCET NUTS-postal code matching system
(https://gisco-services.ec.europa.eu/tercet/flat-files). 

Lists of ITL codes for the United Kingdom can be found here:
        https://geoportal.statistics.gov.uk/documents/international-territorial-levels-level-3-january-2021-names-and-codes-in-the-united-kingdom-v2/about
      	https://geoportal.statistics.gov.uk/datasets/ons::international-territorial-levels-level-2-january-2021-names-and-codes-in-the-united-kingdom/explore
      	https://geoportal.statistics.gov.uk/datasets/ons::international-territorial-levels-level-1-january-2021-names-and-codes-in-the-united-kingdom/explore
Using the respective region names, we manually mapped the regions 1:1 to the corresponding NUTS-3
(and NUTS-2) codes, for both the NUTS-2016 and NUTS-2021 classification scheme. 
