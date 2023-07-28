"""
Downloads data files, mostly from eurostat.

Dependencies
------------
openpyxl

Methods
-------
progress: generator function for output of downloading progress
download_xlsx_to_csv: downloads an xlsx and coverts one sheet to a csv
download_unzip: downloads a gz and unpacks it
download_all: downloads all files for given urls

Exceptions
----------
ConnectionResetError
    might be thrown when the data servers are rejecting the connection, in which case the script needs to be restarted
"""
import os
import argparse
import csv
import urllib.request
import gzip
import shutil
import tempfile

import openpyxl


def progress(totalNo):
    """
    Generator function for output of progress, total numer of the intended calls must be given.

    Throws StopIteration if called more than totalNo times.

    Parameters
    ----------
    totalNo: int
        Maximum number of allowed calls to function

    Yields
    -------
    None
    """
    i = 1
    while i < totalNo:
        print("\rDownloading {} / {}".format(i, totalNo), end='')
        yield
        i += 1
    print("\rDownloading {} / {}".format(i, totalNo))
    yield


def download_xlsx_to_csv(url, filename, sheetNumber):
    """
    Downloads an xlsx from url, saves to tmp file, converts the given sheetNumber to csv and saves in filename.

    Parameters
    ----------
    url: str
        the url of the xlsx file
    filename: str
        the name of the writen csv file
    sheetNumber: int
        the number of the sheet in the xlsx to be saved as csv
    """
    fd, filepath = tempfile.mkstemp(suffix=".xlsx")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'XYZ/3.0'})) as response, \
                open(filepath, 'wb') as tmpfile:
            shutil.copyfileobj(response, tmpfile)
        sheet = openpyxl.load_workbook(filepath, read_only=True).worksheets[sheetNumber]
        with open(filename, "w", newline='') as outfile:
            writer = csv.writer(outfile, delimiter=';')
            for row in sheet:
                values = (cell.value for cell in row)
                writer.writerow(values)
    finally:
        os.remove(filepath)


def download_unzip(url, filename):
    """
    Downloads a gz file, uncompresses it, and saves to filename.

    Parameters
    ----------
    url: str
        the url of the gz file
    filename: str
        the name of the file to be writen
    """
    with urllib.request.urlopen(url) as response, gzip.GzipFile(fileobj=response) as uncompressed:
        with open(filename, 'wb') as outfile:
            shutil.copyfileobj(uncompressed, outfile)
    
                
def download_all(dataDir, legacy):
    """
    Downloads all data to dataDir. If legacy, data for NUTS2016 definitions is downloaded, where applicable.

    dataDir must not exist beforehand and will be created.

    Parameters
    ----------
    dataDir: str
        the name of the new directory for all downloaded files
    legacy: bool
        if true, files for NUTS2016 definitions will be downloaded if they exist

    Returns
    -------
    bool
        True if successful
    """
    if os.path.isdir(dataDir):
        print(dataDir + ' already exists, skipping downloads.\n')
        return False
    
    os.mkdir(dataDir)
    if legacy:
        output = progress(43)
    else:
        output = progress(44)

    next(output)
    url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/prc_ppp_ind.tsv.gz"
    download_unzip(url, os.path.join(dataDir, "prc_ppp_ind.tsv"))
    if not legacy:
        next(output)
        url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/nama_10r_3gdp." \
              "tsv.gz"
        download_unzip(url, os.path.join(dataDir, "nama_10r_3gdp.tsv"))
    next(output)
    url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/ert_bil_eur_a.tsv.gz"
    download_unzip(url, os.path.join(dataDir, "ert_bil_eur_a.tsv"))
    next(output)
    url = " https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/demo_r_pjanaggr3." \
          "tsv.gz"
    download_unzip(url, os.path.join(dataDir, "demo_r_pjanaggr3.tsv"))
    next(output)
    url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/tour_occ_nin2.tsv.gz"
    download_unzip(url, os.path.join(dataDir, "tour_occ_nin2.tsv"))
    next(output)
    url = "https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/ilc_peps11.tsv.gz"
    download_unzip(url, os.path.join(dataDir, "ilc_peps11.csv"))

    countries = ['at', 'be', 'bg', 'ch', 'cy', 'cz', 'de', 'dk', 'ee', 'el', 'es', 'fi', 'fr', 'hr', 'hu', 'ie', 'is',
                 'it', 'lt', 'lu', 'lv', 'me', 'mk', 'mt', 'nl', 'no', 'pl', 'pt', 'ro', 'rs', 'se', 'si', 'sk', 'tr',
                 'uk']
    for country in countries:
        next(output)
        url = 'https://ec.europa.eu/eurostat/estat-navtree-portlet-prod/BulkDownloadListing?file=data/avia_par_' \
              + country + '.tsv.gz'
        download_unzip(url, os.path.join(dataDir, 'avia_par_' + country + '.tsv'))

    next(output)
    if legacy:
        url = "https://ec.europa.eu/eurostat/documents/1797762/1797951/Island-regions-NUTS-2016.xlsx"
        download_xlsx_to_csv(url, os.path.join(dataDir, "Island-regions-NUTS-2016.csv"), 1)
    else:
        url = "https://ec.europa.eu/eurostat/documents/1797762/1797951/Island-regions-NUTS-2021.xlsx"
        download_xlsx_to_csv(url, os.path.join(dataDir, "Island-regions-NUTS-2021.csv"), 0)
    next(output)
    url = "https://www.ons.gov.uk/file?uri=%2feconomy%2fgrossdomesticproductgdp%2fdatasets%2f" \
          "regionalgrossdomesticproductallnutslevelregions%2f1998to2019/regionalgrossdomesticproductallitlregions.xlsx"
    download_xlsx_to_csv(url, os.path.join(dataDir, "regionalgrossdomesticproductallitlregions.csv"), 6)
    next(output)
    url = "https://dam-api.bfs.admin.ch/hub/api/dam/assets/19544503/master"
    download_xlsx_to_csv(url, os.path.join(dataDir, "je-e-04-02-06-01.csv"), 0)
    
    return True


def main(dataDir, legacy):
    """
    Downloads external data.

    Parameters
    ----------
    dataDir: str
        the name of the new directory for all downloaded files
    legacy: bool
        if true, files for NUTS2016 definitions will be downloaded if they exist
    """
    if download_all(dataDir, legacy):
        print("All data successfully downloaded.")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataDir', help='directory for downloaded external data files, dir must not exist beforehand')
    parser.add_argument('--legacy', help='if given, use NUTS2016 definitions', action='store_true', default=False)

    args = parser.parse_args()
    main(args.dataDir, args.legacy)    
