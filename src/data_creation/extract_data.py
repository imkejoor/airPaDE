"""
For a list of given pseudo airports, collects and writes statistical data for single and pairs of airports.

This file can be exported as a module and contains the class DataCollector.
"""
import os
import argparse
import csv

import download_data


def is_EU_zone(country1, country2):
    """
    Returns True if both countries are in the internal EU list, false otherwise.

    Parameters
    ----------
    country1: str
        two-letter, upper case abbreviation of the first country
    country2: str
        two-letter, upper case abbreviation of the second country

    Returns
    -------
    bool
    """
    EU_list = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'IE', 'IT', 'HR',
               'HU', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK']
    if country1 not in EU_list:
        return False
    if country2 not in EU_list:
        return False
    return True


def has_same_currency(country1, country2):
    """
    Returns True if both countries are in the internal EURO list or are identical, False otherwise.

    Parameters
    ----------
    country1: str
        two-letter, upper case abbreviation of the first country
    country2: str
        two-letter, upper case abbreviation of the second country

    Returns
    -------
    bool
    """
    EURO_list = ['AT', 'BE', 'CY', 'DE', 'EE', 'EL', 'ES', 'FI', 'FR', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL',
                 'PT', 'SI', 'SK']
    if country1 == country2:
        return True
    if country1 not in EURO_list:
        return False
    if country2 not in EURO_list:
        return False
    return True


def add_NA_aware(value1, value2):
    """
    Returns a str value of the float addition of value1 and value2, handles input 'NA' like nan.

    Parameters
    ----------
    value1: str
    value2: str

    Returns
    -------
    str
    """
    if value1 == 'NA' or value2 == 'NA':
        return 'NA'
    else:
        return str(float(value1) + float(value2))


def div_NA_aware(value1, value2):
    """
    Returns a str value of the float division of value1 and value2, handles input 'NA' like nan.

    Parameters
    ----------
    value1: str, float, int
    value2: str, float, int

    Returns
    -------
    str
    """
    if value1 == 'NA' or value2 == 'NA':
        return 'NA'
    else:
        return str(float(value1) / float(value2))


def multiply_NA_aware(value1, value2):
    """
    Returns a str value of the float multiplication of value1 and value2, handles input 'NA' like nan.

    Parameters
    ----------
    value1: str
    value2: str

    Returns
    -------
    str
    """
    if value1 == 'NA' or value2 == 'NA':
        return 'NA'
    else:
        return str(float(value1) * float(value2))


def or_NA_aware(value1, value2):
    """
    Returns str(value1 or value1), where (1 or 'NA') = 1, (0 or 'NA') = 'NA'.

    Accepts only '0', '1', and 'NA' as input

    Parameters
    ----------
    value1: str
    value2: str

    Returns
    -------
    str
    """
    assert value1 in ['0', '1', 'NA'] and value2 in ['0', '1', 'NA']
    if value1 == '1' or value2 == '1':
        return '1'
    if value1 == '0' and value2 == '0':
        return '0'
    return 'NA'


class DataCollector:
    """
    Holds and collects data for all airports and computes values for all pairwise combinations of airports.

    Methods
    -------
    collect
        builds airport data, statistical data and pairwise statistical data for all airports
    write_stat(filename: str)
        writes previously collected statistical data to file filename
    write_pairwise_stat(filename: str)
        writes previously collected pairwise statistical data to file filename
    write_passenger(filename: str)
        writes the previously collected number of passenger matrix to file
    data_statistics(outfile: str)
        get the number of airports and compute the number of full data points, write them to outfile

    Parameters
    ----------
    baseFile: str
        filename with general airport data
    dataDir: str
        name of the directory with data files that were downloaded
    dataDirExtra: str
        name of the directory with provided data files
    year: str
        the year for which data should be (ideally) fetched
    logFile: str
        name of the logfile for output of warnings
    fullPairwiseData: bool
        if true, a pair of airports is included in the pairwise stat data even if there are no passenger data

    Attributes
    ----------
    baseFile: str
        filename with general airport data
    dataDir: str
        name of the directory with data files that were downloaded
    dataDirExtra: str
        name of the directory with provided data files
    year: str
        the year for which data should be (ideally) fetched
    logFile: str
        name of the logfile for output of warnings
    fullPairwise: bool
        if true, a pair of airports is included in the pairwise stat data even if there are no passenger data
    airports: dict[str, dict]
        {airport id: {ICAO: , NUTS2: , NUTS3: , country: }}
    icao2id: dict[str, int]
        {icao: id} for airports
    passenger: list[list[int]]
        matrix of the passenger data, indexed by the respective airport ids-1
    statData: list[dict]
        list of dicts; indexed by airport ids-1, contains all external airport data
    statPairwise: list[dict]
        list of dicts; contains one dict with data for every pair of airports with passenger > 0 between them
    GBP2EUR: float
        exchange rate from GBP to Euro
    CHF2EUR: float
        exchange rate from CHF to Euro
    """
    def __init__(self, baseFile, dataDir, dataDirExtra, year, logFile, fullPairwiseData):
        self.baseFile = baseFile
        self.dataDir = dataDir
        self.dataDirExtra = dataDirExtra
        self.year = year
        self.logFile = logFile
        self.fullPairwise = fullPairwiseData
        self.airports = {}
        self.icao2id = {}
        self.passenger = [[]]
        self.statData = [{}]
        self.statPairwise = []
        self.GBP2EUR = 1.2  # will be updated for specific year
        self.CHF2EUR = 0.98

    def collect(self):
        """
        Builds airport data, statistical data and pairwise statistical data for all airports.
        """
        with open(self.logFile, 'a') as f:
            f.write('Collected data is for the year ' + self.year + ', unless otherwise stated:\n\n')
            
        self.build_airports()
        self.build_passenger()
        self.build_stat_data()
        self.build_pairwise_data(os.path.join(self.dataDirExtra, 'dist-data.csv'))

    def build_stat_data(self):
        """
        Builds the data for statData.
        """
        self.statData = [{} for _ in range(len(self.airports))]
        self.get_priceindex(os.path.join(self.dataDir, 'prc_ppp_ind.tsv'))
        self.get_gdp(os.path.join(self.dataDir, 'nama_10r_3gdp.tsv'),
                     os.path.join(self.dataDir, 'ert_bil_eur_a.tsv'),
                     os.path.join(self.dataDir, 'regionalgrossdomesticproductallitlregions.csv'),
                     os.path.join(self.dataDirExtra, 'UK_NUTS3_to_ITL3.csv'),
                     os.path.join(self.dataDirExtra, 'CH_cantons_to_NUTS3.csv'),
                     os.path.join(self.dataDir, 'je-e-04-02-06-01.csv'))
        self.get_pop(os.path.join(self.dataDir, 'demo_r_pjanaggr3.tsv'))
        self.get_nights(os.path.join(self.dataDir, 'tour_occ_nin2.tsv'))
        self.get_geographics(os.path.join(self.dataDirExtra, 'Coastal-regions-NUTS-2021_amended.csv'),
                             os.path.join(self.dataDir, 'Island-regions-NUTS-2021.csv'))
        self.get_poverty(os.path.join(self.dataDir, 'ilc_peps11.csv'))

    def get_priceindex(self, filename):
        """
        Sets the entry PLI NUTS-1 in statData.

        Parameters
        ----------
        filename : str
            file location of the price index data
        """
        price = {}
        with open(filename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            # the column header for the year might have an extra blank space at the end
            try:
                column = header.index(self.year)
            except ValueError:
                column = header.index(self.year + ' ')

            for row in (row for row in reader if row[0].startswith('PLI_EU27_2020,E011,')):
                country = (row[0].split(','))[2]
                if len(country) == 2:
                    price[country] = row[column]

        for i, entry in enumerate(self.statData):
            entry["PLI NUTS-1"] = price[self.airports[str(i+1)]['country']].strip()

    def get_gdp(self, gdpFilename, exFilename, ukgdpFilename, itlFilename, cantonFilename, chgdpFilename):
        """
        Sets the entry GDP in statData for given year, but tries all older values if data is missing.

        Parameters
        ----------
        gdpFilename: str
            filename for GDP data except CH and UK
        exFilename: str
            filename for year dependent exchange rates
        ukgdpFilename: str
            filename for GDP data in UK, given for ITL3 codes
        itlFilename: str
            filename for map from ITL3 to NUTS3 codes
        cantonFilename: str
            filename for map from CH cantons to NUTS3 codes
        chgdpFilename: str
            filename for GDP data in CH, given for cantons
        """
        gdp_nuts3 = {}
        with open(gdpFilename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            try:  # the column header for the year might have an extra blank space at the end
                column = header.index(self.year)
            except ValueError:
                column = header.index(self.year + ' ')

            # take value from year, otherwise try older values, but set entry in logfile
            for row in (row for row in reader if row[0].startswith('MIO_EUR,')):
                nuts3 = (row[0].split(','))[1]
                if len(nuts3) == 5 and not nuts3.startswith('CH'):  # for CH, we have better data, see below
                    if row[column].split(' ')[0] != ':':  # number might have a char for a footnote at the end
                        gdp_nuts3[nuts3] = row[column].split(' ')[0]
                    else:
                        gdp_nuts3[nuts3] = ':'
                        col = column+1
                        while col < len(header):
                            if row[col].split(' ')[0] != ':': 
                                gdp_nuts3[nuts3] = row[col].split(' ')[0]
                                with open(self.logFile, 'a') as f:
                                    f.write(nuts3 + ': gdp data from ' + header[col] +
                                            ' since no newer value (:) in file ' + gdpFilename + '\n')
                                break
                            col += 1
                        
        # extra steps for UK and CH
        self.get_exchange(exFilename)
        self.get_gdp_UK(gdp_nuts3, ukgdpFilename, itlFilename)
        self.get_gdp_CH(gdp_nuts3, cantonFilename, chgdpFilename)

        for i, entry in enumerate(self.statData):
            gdp = 0
            for nuts3 in self.airports[str(i+1)]['NUTS-3']:
                try:
                    gdp += float(gdp_nuts3[nuts3]) 
                except KeyError:  # nuts3 code does not exist in file
                    with open(self.logFile, 'a') as f:
                        f.write(self.airports[str(i+1)]['ICAO'][0] + ' GDP missing, since ' + nuts3 + ' not in file '
                                + gdpFilename + '\n')
                    gdp = 'NA'
                    break
                except ValueError:
                    if gdp_nuts3[nuts3] == ':':  # no value given
                        with open(self.logFile, 'a') as f:
                            f.write(self.airports[str(i+1)]['ICAO'][0] + ': GDP missing, since ' + nuts3
                                    + ' has no value (:) in file ' + gdpFilename + '\n')
                        gdp = 'NA'
                        break
            entry["GDP NUTS-3"] = gdp

    def get_exchange(self, filename):
        """
        Look up exchange rates for GBP and CHF and set variables GBP2EUR and CHF2EUR.

        Parameters
        ----------
        filename: str
            filename for exchange rates
        """
        with open(filename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            try:
                column = header.index(self.year)
            except ValueError:
                column = header.index(self.year + ' ')
           
            for row in reader:
                if row[0].startswith('AVG,NAC,GBP'):
                    self.GBP2EUR = float(row[column])
                if row[0].startswith('AVG,NAC,CHF'):
                    self.CHF2EUR = float(row[column])
                    
    def get_gdp_UK(self, gdp_nuts3, ukgdpFilename, itlFilename):
        """
        Read GDP data for ITL3 codes, look up NUTS3 codes for ITL3, and set entry GDP for NUTS3.

        Parameters
        ----------
        gdp_nuts3: dict
            dictionary for GDP values indexed by NUTS3 codes
        ukgdpFilename: str
            filename for GDP data for ITL3 codes, given in GBP
        itlFilename: str
            filename for map from ITL3 to NUTS3 codes
        """
        # look up gdp for itl3 code (given in GBP)
        itl3_gdp = {}
        with open(ukgdpFilename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter=';')
            next(reader)  # this file has an extra line at the beginning
            header = next(reader)
            try:
                column = header.index(self.year)
            except ValueError:  # probably some weird additional content in cell
                for i, item in enumerate(header):
                    if item[:4] == self.year:
                        column = i
                        break
                else:
                    raise ValueError("Requested year not found in regionalgrossdomesticproduct")
            
            for row in (row for row in reader if len(row[1].strip()) == 5):
                itl3_gdp[row[1].strip()] = float(row[column])

        # look up nuts3 list and put it together in gdp for nuts3 in EUR
        with open(itlFilename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter=';')
            next(reader)
            for row in (row for row in reader if len(row[0].strip()) == 5):  # this is a nuts3 instead of nuts2 code
                gdp_nuts3[row[0].strip()] = itl3_gdp[row[1].strip()] * self.GBP2EUR

    def get_gdp_CH(self, gdp_nuts3, cantonFilename, chgdpFilename):
        """
        Read GDP data for cantons from file, read map from cantons to NUTS3 codes, and set entry GDP for NUTS3.

        Parameters
        ----------
        gdp_nuts3: dict
            dictionary for GDP values indexed by NUTS3 codes
        cantonFilename: str
            filename for GDP data for cantons, given in CHF
        chgdpFilename: str
            filename for map from cantons to NUTS3 codes
        """
        with open(cantonFilename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter=';')
            nuts3_canton = {row[0]: row[1] for row in reader}

        with open(chgdpFilename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter=';')
            next(reader)  # this file has two extra lines at the beginning
            next(reader)
            header = next(reader)
            try:
                column = header.index(self.year)
            except ValueError:  # additional comment on year maybe
                for i, item in enumerate(header):
                    if item[:4] == self.year:
                        column = i
                        break
                else:
                    raise ValueError("Requested year not found in je-e-04.02.06.01.csv")

            # relevant lines are between 'In CHF' and 'Switzerland'
            row = next(reader)  # and one more line with stuff
            while not row[0].startswith('In CHF million'):
                row = next(reader)
            row = next(reader)
            while not row[0].startswith('Switzerland'):
                gdp_nuts3[nuts3_canton[row[0]]] = float(row[column]) * self.CHF2EUR
                row = next(reader)

    def get_pop(self, filename):
        """
        Sets entries Pop NUTS-3 and Pop NUTS-2 in statData for given year, but tries year-1 if data is missing.

        Parameters
        ----------
        filename: str
            filename for population data
        """
        pop_nuts2 = {}
        pop_nuts3 = {}
        with open(filename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            try:  # the column header for the year might have an extra blank space at the end
                column = header.index(self.year)
            except ValueError:
                column = header.index(self.year + ' ')

            for row in (row for row in reader if row[0].startswith('NR,T,TOTAL,')):
                nuts = (row[0].split(','))[3]
                if len(nuts) == 4:
                    pop_nuts2[nuts] = row[column]
                if len(nuts) == 5:
                    pop_nuts3[nuts] = row[column]
                    if pop_nuts3[nuts] == ': ' and row[column+1] != ': ':  # no number for this year, try one before
                        pop_nuts3[nuts] = row[column+1]
                        with open(self.logFile, 'a') as f:
                            f.write('{}: population data from {} since no newer value (:) in file '
                                    '{}\n'.format(nuts, int(self.year)-1, filename))
            
        for i, entry in enumerate(self.statData):
            pop2 = 0
            for nuts2 in self.airports[str(i+1)]['NUTS-2']:
                try:
                    pop2 += float(pop_nuts2[nuts2].split(' ')[0])  # number might have a char for a footnote at the end
                except ValueError:  # no value for nuts2
                    entry["Pop NUTS-2"] = 'NA'
                    break
            entry["Pop NUTS-2"] = str(pop2)

            pop3 = 0
            for nuts3 in self.airports[str(i+1)]['NUTS-3']:
                try:
                    pop3 += float(pop_nuts3[nuts3].split(' ')[0])  # number might have a char for a footnote at the end
                except ValueError:  # no value for nuts3
                    entry["Pop NUTS-3"] = 'NA'
                    break
            entry["Pop NUTS-3"] = str(pop3)

    def get_nights(self, filename):
        """
        Sets the entry Nights NUTS-2 in statData for given year, but tries all earlier years if data is missing.

        Parameters
        ----------
        filename: str
            filename for data nights spent in NUTS2 area
        """
        nights_nuts2 = {}
        with open(filename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            column = header.index(self.year + ' ')

            for row in (row for row in reader if row[0].startswith('TOTAL,NR,I551-I553,')):
                nuts2 = (row[0].split(','))[3]
                if len(nuts2) != 4:  # this is actually not a NUTS2 code
                    continue
                # a lot of missing data, so start with the newest ones and go backwards
                col = column
                while col < len(header):
                    if row[col].split(' ')[0] != ':':
                        nights_nuts2[nuts2] = row[col].split(' ')[0]
                        break
                    col += 1
                else:  # passed last column, so no data in any column
                    with open(self.logFile, 'a') as f:
                        f.write('{}: no night data down to year {} in file {}\n'.format(nuts2, header[col-1],
                                                                                         filename))
            
        for i, entry in enumerate(self.statData):
            nights = 0
            for nuts2 in self.airports[str(i+1)]['NUTS-2']:
                try:
                    nights += float(nights_nuts2[nuts2])
                except KeyError:  # nuts2 code does not exist in file, and hence, in nights_nuts2
                    with open(self.logFile, 'a') as f:
                        f.write('For {}, nights missing, since {} not in file {}'
                                '\n'.format(self.airports[str(i+1)]['ICAO'][0], nuts2, filename))
                    nights = 'NA'
                    break
            entry["Nights NUTS-2"] = str(nights)

    def get_geographics(self, coastalFilename, islandsFilename):
        """
        Sets the entries Coastal NUTS-3 and Island NUTS-3 in statData.

        Parameters
        ----------
        coastalFilename: str
            filename for coastal regions
        islandsFilename: str
            filename for islands
        """
        coastal_nuts3 = {}
        with open(coastalFilename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter=',')
            next(reader)
            for row in reader:
                coastal_nuts3[row[0]] = row[1]  # second column contains 'Y' or 'N'

        islands = []
        with open(islandsFilename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter=';')
            next(reader)
            for row in reader:
                islands.append(row[0])

        for i, entry in enumerate(self.statData):
            coastal = 0
            island = 0
            for nuts3 in self.airports[str(i+1)]['NUTS-3']:
                try:
                    if coastal_nuts3[nuts3] == 'Y':
                        coastal = 1
                except KeyError:  # nuts3 does not exist in file 'Coastal-regions'
                    with open(self.logFile, 'a') as f:
                        f.write(self.airports[str(i+1)]['ICAO'][0] + ' coastal region missing, since ' + nuts3
                                + ' not in file ' + coastalFilename + '\n')
                        coastal = 'NA'
                if nuts3 in islands:
                    island = 1

            entry["Coastal NUTS-3"] = str(coastal)
            entry["Island NUTS-3"] = str(island)

    def get_poverty(self, filename):
        """
        Sets the entries for Poverty percentage in statDat for given year.

        Parameters
        ----------
        filename: str
            filename for the percentage of people at risk of poverty or social exclusion for countries
        """
        pov_country = {}
        with open(filename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            try:
                column = header.index(self.year)
            except ValueError:
                column = header.index(self.year + ' ')
            # look only for countries, i.e., no numbers in descriptor
            for row in (row for row in reader if row[0].startswith('PC,') and len(row[0].split(',')[1]) == 2):
                if row[0].split(',')[1] == 'XK':      # Kosovo is otherwise not part of the data set
                    continue
                if row[column].split(' ')[0] != ':':  # number might have a char for a footnote at the end
                    pov_country[row[0].split(',')[1]] = float(row[column].split(' ')[0])
                else:
                    col = column+1
                    while col < len(header):
                        if row[col].split(' ')[0] != ':': 
                            pov_country[row[0].split(',')[1]] = float(row[col].split(' ')[0])
                            with open(self.logFile, 'a') as f:
                                f.write('{}: poverty data from {} since no newer value (:) in file {}'
                                        '\n'.format(row[0].split(',')[1], header[col], filename))
                            break
                        col += 1
                    else:
                        with open(self.logFile, 'a') as f:
                            f.write('{}: no poverty data since there are no values older or equal to {} in file {}'
                                    '\n'.format(row[0].split(',')[1], self.year, filename))
                        pov_country[row[0].split(',')[1]] = 'NA'
        for i, entry in enumerate(self.statData):
            entry["Poverty percentage"] = div_NA_aware(pov_country[self.airports[str(i+1)]['country']], 100)

    def build_pairwise_data(self, filename):
        """
        Sets all data in statPairwise.

        Parameters
        ----------
        filename: str
            filename for distance matrix between all airports
        """
        for i in range(len(self.airports)-1):
            for j in range(i+1, len(self.passenger)):
                num = max(self.passenger[i][j], self.passenger[j][i])
                # an entry should be set if fullPairwise is requested, otherwise only if there are passenger data
                if self.fullPairwise or num > 0:
                    self.build_pairwise_entry(i, j, num)

        with open(filename, newline='') as inFile:
            dialect = csv.Sniffer().sniff(inFile.read(1024))
            inFile.seek(0)
            reader = csv.reader(inFile, dialect)
            dist = list(reader)
        for pair in self.statPairwise:
            pair["dist"] = dist[int(pair["i"])-1][int(pair["j"])-1]

    def build_pairwise_entry(self, i, j, num):
        """
        Sets entry for airport pair (i,j) in statPairwise, uses previously build statData to compute sums and products.

        Parameters
        ----------
        i: int
            index of first airport
        j: int
            index of second airport
        num: int
            number of passengers between i and j
        """
        entry = {"i": str(i + 1), "j": str(j + 1), "PAX": num if num > 0 else 'NA'}

        country1 = self.airports[entry["i"]]["country"]
        country2 = self.airports[entry["j"]]["country"]
        entry["domestic"] = int(country1 == country2)
        entry["inter-EU"] = int(is_EU_zone(country1, country2))
        entry["same currency"] = int(has_same_currency(country1, country2))

        entry["PLI prod"] = multiply_NA_aware(self.statData[i]["PLI NUTS-1"], self.statData[j]["PLI NUTS-1"])
        entry["PLI sum"] = add_NA_aware(self.statData[i]["PLI NUTS-1"], self.statData[j]["PLI NUTS-1"])
        entry["GDP prod"] = multiply_NA_aware(self.statData[i]["GDP NUTS-3"], self.statData[j]["GDP NUTS-3"])
        entry["GDP sum"] = add_NA_aware(self.statData[i]["GDP NUTS-3"], self.statData[j]["GDP NUTS-3"])

        entry["population prod"] = multiply_NA_aware(self.statData[i]["Pop NUTS-3"], self.statData[j]["Pop NUTS-3"])
        entry["population sum"] = add_NA_aware(self.statData[i]["Pop NUTS-3"], self.statData[j]["Pop NUTS-3"])
        entry["catchment prod"] = multiply_NA_aware(self.statData[i]["Pop NUTS-2"], self.statData[j]["Pop NUTS-2"])
        entry["catchment sum"] = add_NA_aware(self.statData[i]["Pop NUTS-2"], self.statData[j]["Pop NUTS-2"])

        entry["nights prod"] = multiply_NA_aware(self.statData[i]["Nights NUTS-2"], self.statData[j]["Nights NUTS-2"])
        entry["nights sum"] = add_NA_aware(self.statData[i]["Nights NUTS-2"], self.statData[j]["Nights NUTS-2"])
        entry["coastal OR"] = or_NA_aware(self.statData[i]["Coastal NUTS-3"], self.statData[j]["Coastal NUTS-3"])
        entry["island OR"] = or_NA_aware(self.statData[i]["Island NUTS-3"], self.statData[j]["Island NUTS-3"])

        entry["poverty prod"] = multiply_NA_aware(self.statData[i]["Poverty percentage"],
                                                  self.statData[j]["Poverty percentage"])
        entry["poverty sum"] = add_NA_aware(self.statData[i]["Poverty percentage"],
                                            self.statData[j]["Poverty percentage"])
        self.statPairwise.append(entry)

    def build_airports(self):
        """
        Sets airports by reading data from baseFile, and icao2id.
        """
        with open(self.baseFile, newline='') as inFile:
            dialect = csv.Sniffer().sniff(inFile.read(1024))
            inFile.seek(0)
            reader = csv.reader(inFile, dialect) 
            header = next(reader)

            idx = [header.index("ID"), header.index("ICAO codes"), header.index("NUTS-3 aggregated"),
                   header.index("NUTS-2 localizations")]

            for row in reader:
                data = {"ICAO": row[idx[1]].split(','), "NUTS-2": row[idx[3]].split(','),
                        "NUTS-3": row[idx[2]].split(',')}
                # read out country code as the first two letters of some NUTS-2 code
                data["country"] = data["NUTS-2"][0][:2]
                self.airports[row[idx[0]]] = data
                # build icao2id; index is then 0-based for passenger matrix
                for icao in data["ICAO"]:
                    self.icao2id[icao] = int(row[idx[0]])-1

    def build_passenger(self):
        """
        Sets passenger, by adding up the values in all avia files in dataDir.
        """
        self.passenger = [[0 for _ in range(len(self.airports))] for _ in range(len(self.airports))]
        
        with os.scandir(self.dataDir) as direc:
            for f in (f for f in direc if (f.is_file() and f.name.startswith('avia_par'))):
                # noinspection PyTypeChecker
                self.read_data(f)

    def read_data(self, f):
        """
        Updates passenger matrix by values from f for given year, missing data (':') are ignored.

        Parameters
        ----------
        f: os.DirEntry, str
            file path or filename for passenger data
        """
        with open(f, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            # the column header for the year might have an extra blank space at the end
            try:
                column = header.index(self.year)
            except ValueError:
                try:
                    column = header.index(self.year + ' ')
                except ValueError:
                    with open(self.logFile, 'a') as outfile:
                        outfile.write('For country {}, no passenger data for {} in file {}'
                                      '\n'.format((f.name.split('_')[2]).split('.')[0].upper(), self.year, f.name))
                    return

            for row in reader:
                # the values we want are only in the rows PAS,PAS_BRD,country_airport1_country_airport2
                if row[0].startswith('PAS,PAS_BRD,'):
                    _, port1, _, port2 = (row[0].split(',')[2]).split('_')
                    if port1 in self.icao2id and port2 in self.icao2id and row[column] != ': ':
                        self.passenger[self.icao2id[port1]][self.icao2id[port2]] += int(row[column].split(' ')[0])
                    
    def write_stat(self, filename):
        """
        Writes statData to csv file.

        Parameters
        ----------
        filename: str
            filename to which data is writen, should be .csv
        """
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["Pop NUTS-3", "Pop NUTS-2", "GDP NUTS-3", "PLI NUTS-1", "Nights NUTS-2", "Coastal NUTS-3",
                          "Island NUTS-3", "Poverty percentage"]
            writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.statData:
                writer.writerow(entry)                

    def write_pairwise_stat(self, filename):
        """
        Writes statPairwise to csv file.

        Parameters
        ----------
        filename: str
            filename to which data is writen, should be .csv
        """
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["i", "j", "dist", "domestic", "inter-EU", "same currency", "PAX",
                          "population prod", "population sum", "catchment prod", "catchment sum", "GDP prod", "GDP sum",
                          "PLI prod", "PLI sum", "nights prod", "nights sum", "coastal OR", "island OR",
                          "poverty sum", "poverty prod"]
            writer = csv.DictWriter(csvfile, delimiter=';', fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.statPairwise:
                writer.writerow(entry)

    def write_passenger(self, filename):
        """
        Writes passenger matrix to csv file.

        Parameters
        ----------
        filename: str
            filename to which data is writen, should be .csv
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for entry in self.passenger:
                writer.writerow(entry)

    def data_statistics(self, outfile):
        """
        Computes and outputs number of individual airports in pairwise data and number of full pairwise data points.

        Parameters
        ----------
        outfile: str
            filename to which the numbers are writen
        """
        airports = set()
        noFullEntries = 0
        for entry in self.statPairwise:
            if 'NA' not in entry.values():
                airports.add(entry['i'])
                airports.add(entry['j'])
                noFullEntries += 1

        with open(outfile, 'a') as f:
            f.write('\nData statistics:\n{} individual airports in\n'.format(len(airports)))
            f.write('{} full data points,\n'.format(noFullEntries))
            f.write('{} data points in total\n'.format(len(self.statPairwise)))

class DataCollectorTransition(DataCollector):
    """
    Holds and collects data for all airports and computes values for all pairwise combinations of airports.

    Derived from DataCollector to handle transition phase, where NUTS2021 codes are used in general, but population and
    nights spent data are only available in NUTS2016 codes.
    Uses an additionally provided file with translation from NUTS2021 -> NUTS2016 definitions for NUTS3 wherever
    this is possible.

    Attributes
    ----------
    nuts21_nuts16: dict[str, list[str]]
        {NUTS16 code: list of all corresponding NUTS21 codes} for all NUTS3 regions that had changes
    no_old_nuts2: list[str]
        list of all NUTS2 codes that do not exist anymore and must be excluded
    """
    def __init__(self, baseFile, dataDir, dataDirExtra, year, logFile, fullPairwiseData):
        """
        Extends super method by additional attributes nuts21_nuts16 and no_old_nuts2.
        """
        super().__init__(baseFile, dataDir, dataDirExtra, year, logFile, fullPairwiseData)
        self.nuts21_nuts16 = {}
        self.no_old_nuts2 = []

    def build_stat_data(self):
        """
        Builds the data for statData.

        Overrides superclass method. Partly uses different data files and calls additional lookup for NUTS codes.
        """
        self.statData = [{} for _ in range(len(self.airports))]
        self.get_priceindex(os.path.join(self.dataDir, 'prc_ppp_ind.tsv'))
        self.get_gdp(os.path.join(self.dataDir, 'nama_10r_3gdp.tsv'),
                     os.path.join(self.dataDir, 'ert_bil_eur_a.tsv'),
                     os.path.join(self.dataDir, 'regionalgrossdomesticproductallitlregions.csv'),
                     os.path.join(self.dataDirExtra, 'UK_NUTS3_to_ITL3.csv'),
                     os.path.join(self.dataDirExtra, 'CH_cantons_to_NUTS3.csv'),
                     os.path.join(self.dataDir, 'je-e-04-02-06-01.csv'))
        self.build_lookup_nutscodes(os.path.join(self.dataDirExtra, 'NUTS2016to2021recodings.csv'))
        self.get_pop(os.path.join(self.dataDir, 'demo_r_pjanaggr3.tsv'))
        self.get_nights(os.path.join(self.dataDir, 'tour_occ_nin2.tsv'))
        self.get_geographics(os.path.join(self.dataDirExtra, 'Coastal-regions-NUTS-2021_amended.csv'),
                             os.path.join(self.dataDir, 'Island-regions-NUTS-2021.csv'))
        self.get_poverty(os.path.join(self.dataDir, 'ilc_peps11.csv'))

    def build_lookup_nutscodes(self, filename):
        """
        Sets lookup nuts21_nuts16 from NUTS2021 to NUTS2016 codes and list no_old_nuts2 of non-existing nuts2 codes.
        Parameters
        ----------
        filename: str
            file location for csv with code translation
        """
        with open(filename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            next(reader)
            for row in reader:
                self.nuts21_nuts16[row[0]] = row[1].split(',')
                if row[3] == 'N/A':
                    self.no_old_nuts2.append(row[2])
            
    def get_pop(self, filename):
        """
        Sets entries Pop NUTS-3 and Pop NUTS-2 in statData for given year, but tries year-1 if data is missing.

        Replaces super method. Additionally ignores missing NUTS2 codes according to no_old_nuts2 and translates NUTS3
        codes in NUTS2016 definition (as in filename) to NUTS2021 definition (as in airport declarations) according to
        nuts21_nuts16.

        Parameters
        ----------
        filename: str
            filename for population data
        """
        pop_nuts2 = {}
        pop_nuts3 = {}
        with open(filename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            # the column header for the year might have an extra blank space at the end
            try:
                column = header.index(self.year)
            except ValueError:
                column = header.index(self.year + ' ')

            for row in (row for row in reader if row[0].startswith('NR,T,TOTAL,')):
                nuts = (row[0].split(','))[3]
                if len(nuts) == 4:
                    pop_nuts2[nuts] = row[column].split(' ')[0]  # number might have a char for a footnote at the end
                if len(nuts) == 5:
                    pop_nuts3[nuts] = row[column].split(' ')[0]
                    if pop_nuts3[nuts] == ':' and row[column+1] != ': ':  # no number for this year, try one before
                        pop_nuts3[nuts] = row[column+1].split(' ')[0]  # number might have a char for footnote at end
                        with open(self.logFile, 'a') as f:
                            f.write('{}: population data from {} since no newer value (:) in file {}'
                                    '\n'.format(nuts, int(self.year)-1, filename))
            
        for i, entry in enumerate(self.statData):
            pop2 = 0
            for nuts2 in self.airports[str(i+1)]['NUTS-2']:
                if nuts2 in self.no_old_nuts2:  # we cannot find this code in the read data, so set to 'NA'
                    with open(self.logFile, 'a') as f:
                        f.write('For {}, population data NUTS2 missing, since {} does not exist in NUTS2016 definition'
                                '\n'.format(self.airports[str(i+1)]['ICAO'][0], nuts2))
                    pop2 = 'NA'
                    break
                pop2 += float(pop_nuts2[nuts2]) 
            entry["Pop NUTS-2"] = str(pop2)

            pop3 = 0
            for nuts3 in self.airports[str(i+1)]['NUTS-3']:
                if nuts3 in self.nuts21_nuts16:  # NUTS3 code was renamed and possibly merged
                    pop3 += sum(float(pop_nuts3[x]) for x in self.nuts21_nuts16[nuts3])
                else:
                    pop3 += float(pop_nuts3[nuts3])
            entry["Pop NUTS-3"] = str(pop3)

    def get_nights(self, filename):
        """
        Sets the entry Nights NUTS-2 in statData for given year, but tries all earlier years if data is missing.

        Replaces super method. Additionally ignores missing NUTS2 codes according to no_old_nuts2 and sets log entry
        accordingly.

        Parameters
        ----------
        filename: str
            filename for data nights spent in NUTS2 area
        """
        nights_nuts2 = {}
        with open(filename, newline='') as inFile:
            reader = csv.reader(inFile, delimiter='\t')
            header = next(reader)
            column = header.index(self.year + ' ')

            for row in (row for row in reader if row[0].startswith('TOTAL,NR,I551-I553,')):
                nuts2 = (row[0].split(','))[3]
                if len(nuts2) != 4:  # this is actually not a NUTS2 code
                    continue
                # a lot of missing data, so start with the newest ones and go backwards
                col = column
                while col < len(header):
                    if row[col].split(' ')[0] != ':':
                        nights_nuts2[nuts2] = row[col].split(' ')[0]
                        break
                    col += 1
                else:  # passed last column, so no data in any column
                    with open(self.logFile, 'a') as f:
                        f.write('{}: no night data down to year {} in file {}'
                                '\n'.format(nuts2, header[col-1], filename))
            
        for i, entry in enumerate(self.statData):
            nights = 0
            for nuts2 in self.airports[str(i+1)]['NUTS-2']:
                try:
                    nights += float(nights_nuts2[nuts2])
                except KeyError:  # nuts2 code does not exist in file, and hence, in nights_nuts2
                    with open(self.logFile, 'a') as f:
                        if nuts2 in self.no_old_nuts2:
                            f.write('For {}, nights missing, since {} does not exist in NUTS2016 definition'
                                    '\n'.format(self.airports[str(i+1)]['ICAO'][0], nuts2))
                        else:
                            f.write('For {}, nights missing, since {} not in file {}'
                                    '\n'.format(self.airports[str(i+1)]['ICAO'][0], nuts2, filename))
                    nights = 'NA'
                    break
            entry["Nights NUTS-2"] = str(nights)
   
            
class DataCollectorLegacy(DataCollector):
    """
    Holds and collects data for all airports and computes values for all pairwise combinations of airports.

    Derived from DataCollector to handle older data (before 2019) with NUTS2016 definition codes, which results in
    slightly different input files.
    """

    def collect(self):
        """
        Builds airport data, statistical data and pairwise statistical data for all airports.

        Overrides superclass method; uses different dist-data file.
        """
        with open(self.logFile, 'a') as f:
            f.write('Collected data is for the year ' + self.year + ', unless otherwise stated:\n\n')
            
        self.build_airports()
        self.build_passenger()
        self.build_stat_data()
        self.build_pairwise_data(os.path.join(self.dataDirExtra, 'dist-data_NUTS2016.csv'))

    def build_stat_data(self):
        """
        Builds the data for statData.

        Overrides superclass method. Partly uses different data files.
        """
        self.statData = [{} for _ in range(len(self.airports))]
        self.get_priceindex(os.path.join(self.dataDir, 'prc_ppp_ind.tsv'))
        self.get_gdp(os.path.join(self.dataDirExtra, 'nama_10r_3gdp_NUTS2016.tsv'),
                     os.path.join(self.dataDir, 'ert_bil_eur_a.tsv'),
                     os.path.join(self.dataDir, 'regionalgrossdomesticproductallitlregions.csv'),
                     os.path.join(self.dataDirExtra, 'UK_NUTS3_to_ITL3_NUTS2016.csv'),
                     os.path.join(self.dataDirExtra, 'CH_cantons_to_NUTS3.csv'),
                     os.path.join(self.dataDir, 'je-e-04-02-06-01.csv'))
        self.get_pop(os.path.join(self.dataDir, 'demo_r_pjanaggr3.tsv'))
        self.get_nights(os.path.join(self.dataDir, 'tour_occ_nin2.tsv'))
        self.get_geographics(os.path.join(self.dataDirExtra, 'Coastal-regions-NUTS-2016_amended.csv'),
                             os.path.join(self.dataDir, 'Island-regions-NUTS-2016.csv'))
        self.get_poverty(os.path.join(self.dataDir, 'ilc_peps11.csv'))
 

def main(baseFile, outFileStat, outFilePax, outFileStatPair, dataDir, dataDirExtra, downloadData, year, legacy,
         dataDirLegacy, transition, logFile, printDataStatistics, fullPairwiseData):
    """
    Checks if external data is there, calls DataCollector class and writes airport and pairwise airports data files.

    Parameters
    ----------
    baseFile: str
        filename with general airport data
    outFileStat: str
        filename for the collected data for single airports
    outFilePax: str
        filename for the collected matrix of passenger data
    outFileStatPair:
        filename for the collected data for pairs of airports
    dataDir: str
        name of the directory with data files that were downloaded
    dataDirExtra: str
        name of the directory with provided data files
    downloadData: bool
        should download_data be called
    year: str
        the year for which data should be (ideally) fetched
    legacy: bool
        should the script run for NUTS2016 definitions; can only be used for years 2018 or older
    dataDirLegacy: str
        name of directory with provided data files for legacy mode
    transition: bool
        should the script run in transition mode; i.e., with NUTS2021 definitions, but some data only from NUTS2016
    logFile: str
        name of the logfile for output of warnings
    printDataStatistics: bool
        should the number of different pseudo airports and complete data points be determined and printed to logfile
    fullPairwiseData: bool
        should an airport pair be included in the outFileStatPair even if passenger data for this pair is missing
    """
    open(logFile, 'w').close()  # wipe logFile clean
    # check input parameters
    if legacy and int(year) > 2018:
        with open(logFile, 'a') as f:
            f.write('Program aborted due to illegitimate parameters.\n')
        exit('Data for NUTS2016 codes is only available up until 2018. Remove legacy flag or set year smaller than 2019'
             ' and run again.')
    if int(year) < 2008:
        with open(logFile, 'a') as f:
            f.write('Program aborted due to illegitimate parameters.\n')
        exit('Not all data is available before 2008, this would crash. '
             'Choose a later year or find older data manually.')

    if downloadData:
        if download_data.download_all(dataDir, legacy):
            with open(logFile, 'a') as f:
                f.write('Downloaded new data to directory ' + dataDir + '.\n')

    if legacy:
        dc = DataCollectorLegacy(baseFile, dataDir, dataDirLegacy, year, logFile, fullPairwiseData)
    elif transition:
        dc = DataCollectorTransition(baseFile, dataDir, dataDirExtra, year, logFile, fullPairwiseData)
    else:
        dc = DataCollector(baseFile, dataDir, dataDirExtra, year, logFile, fullPairwiseData)
    dc.collect()
    if printDataStatistics:
        dc.data_statistics(logFile)

    dc.write_stat(outFileStat)
    dc.write_pairwise_stat(outFileStatPair)   
    # dc.write_passenger(outFilePax) #  we could get rid of matrix and passenger file, but keep it for legacy reasons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--baseFile', help='csv file with airport ids',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'base-data.csv'))
    parser.add_argument('--outFileStat', help='output file for statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'stat-data.csv'))
    parser.add_argument('--outFilePax', help='output file for passenger data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'passengers-on-board-data.csv'))
    parser.add_argument('--outFileStatPair', help='output file for pairwise statistical data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'pairwise-stat-data.csv'))
    parser.add_argument('--dataDir', help='directory for downloaded external data files',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'external_data'))
    parser.add_argument('--dataDirExtra', help='directory for additionally shipped data',
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'additional_data'))
    parser.add_argument('--downloadData', help='if given, external data will be downloaded first',
                        action='store_true', default=False)
    parser.add_argument('--year', help='the base year for the data, corresponds to column in avia files',
                        default='2019')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--legacy', help='if given, NUTS2016 definitions will be used',
                       action='store_true', default=False)
    parser.add_argument('--dataDirLegacy', help='directory for NUTS2016 data',
                        default='legacy_nuts2016')
    group.add_argument('--transition', help='must be used while tour and demo are still using NUTS2016 definitions',
                       action='store_true', default=False)
    parser.add_argument('--logFile', help='file where missing data is reported',
                        default='extract.log')
    parser.add_argument('--printDataStatistics', help='if given, number of full data points will be determined and '
                                                      'printed', action='store_true', default=False)
    parser.add_argument('--fullPairwiseData', help='if given, the output for pairwise-data will include airport pairs '
                                                   'even if passenger data for this pair missing; to be used in a case '
                                                   'where the statistical parameters are only trained',
                        action='store_true', default=False)
    args = parser.parse_args()
    main(args.baseFile, args.outFileStat, args.outFilePax, args.outFileStatPair, args.dataDir, args.dataDirExtra,
         args.downloadData, args.year, args.legacy, args.dataDirLegacy, args.transition, args.logFile,
         args.printDataStatistics, args.fullPairwiseData)
