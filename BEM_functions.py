#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import glob as gb

from eppy import modeleditor
from eppy.modeleditor import IDF

import tqdm

import multiprocessing
import multiprocess as mp

import shlex
import subprocess
from itertools import product

import pathlib
from pathlib import Path

import sqlite3

import pandas as pd
import numpy as np

import datetime


# In[ ]:


ORIGIN_DIR = '.'
IDF_DIR = Path('./files/EnergyPlus/copies').absolute()
OUT_DIR_EPlus = Path('./outputs/energyplus').absolute()


# In[ ]:


gb.glob('**/*.sql', recursive=True)


# In[ ]:


igu_low_perf = ['dg_init_bronze', 'dg_0_clear', 'sg_1_clear', 'sg_2_coated']
igu_high_perf = ['dg_1_highSHG_highLT', 'dg_2_midSHG_midLT',
                 'dg_3_midSHG_highLT', 'dg_4_lowSHG_lowLT',
                 'dg_5_lowSHG_midLT', 'dg_6_lowSHG_highLT',
                 'dg_5k_Krypton_lowSHG_midLT', 'dg_vacuum',
                 'tg_1_highSHG_highLT', 'tg_2_midSHG_midLT',
                 'tg_3_midSHG_highLT', 'tg_4_lowSHG_lowLT',
                 'tg_5_lowSHG_midLT', 'tg_6_lowSHG_highLT',
                 'tg_5k_Krypton_lowSHG_midLT', 'tg_5x_Xenon_lowSHG_midLT',
                 'ccf'
                 ]


# In[1]:


net_conditioned_area = 8100
glazed_facade_area = 2750


# In[ ]:


class EPLusSQL():

    def __init__(self, sql_path=None):
        abs_sql_path = os.path.abspath(sql_path)
        self.sql_uri = '{}?mode=ro'.format(pathlib.Path(abs_sql_path).as_uri())

    def get_annual_energy_by_fuel_and_enduse(self):
        """
        Queries SQL file and returns the ABUPS' End Uses table

        Parameters
        ----------
        None

        Returns
        -------
        df_end_use: pd.DataFrame
            Annual End Use table
            index = 'EndUse'
            columns = ['FuelType','Units']
        """

        # RowName = '#{end_use}'
        # ColumnName='#{fuel_type}'
        annual_end_use_query = """SELECT RowName, ColumnName, Units, Value
            FROM TabularDataWithStrings
            WHERE ReportName='AnnualBuildingUtilityPerformanceSummary'
            AND ReportForString='Entire Facility'
            AND TableName='End Uses'
        """

        with sqlite3.connect(self.sql_uri, uri=True) as con:
            df_end_use = pd.read_sql(annual_end_use_query, con=con)

        # Convert Value to Float
        df_end_use['Value'] = pd.to_numeric(df_end_use['Value'])

        df_end_use = df_end_use.set_index(['RowName',
                                           'ColumnName',
                                           'Units'])['Value'].unstack([1, 2])
        df_end_use.index.name = 'EndUse'
        df_end_use.columns.names = ['FuelType', 'Units']

        end_use_order = ['Heating', 'Cooling',
                         'Interior Lighting', 'Exterior Lighting',
                         'Interior Equipment', 'Exterior Equipment',
                         'Fans', 'Pumps', 'Heat Rejection', 'Humidification',
                         'Heat Recovery', 'Water Systems',
                         'Refrigeration', 'Generators']
        col_order = [
            'Electricity', 'Natural Gas', 'Gasoline', 'Diesel', 'Coal',
            'Fuel Oil No 1', 'Fuel Oil No 2', 'Propane', 'Other Fuel 1',
            'Other Fuel 2', 'District Cooling', 'District Heating',
            'Water']
        df_end_use = df_end_use[col_order].loc[end_use_order]

        # Filter out columns with ALL zeroes
        df_end_use = df_end_use.loc[:, (df_end_use > 0).any(axis=0)]

        return df_end_use

    def get_unmet_hours_table(self):
        """
        Queries 'SystemSummary' and returns all unmet hours

        Parameters
        ----------
        None

        Returns
        -------
        df_unmet: pd.DataFrame
            A DataFrame where


        """

        query = """SELECT RowName, ColumnName, Units, Value FROM TabularDataWithStrings
    WHERE ReportName='SystemSummary'
    AND ReportForString='Entire Facility'
    AND TableName='Time Setpoint Not Met'
    """
        with sqlite3.connect(self.sql_uri, uri=True) as con:
            df_unmet = pd.read_sql(query, con=con)

        # Convert Value to Float
        df_unmet['Value'] = pd.to_numeric(df_unmet['Value'])

        df_unmet = df_unmet.pivot(index='RowName',
                                  columns='ColumnName',
                                  values='Value')

        df_unmet.columns.names = ['Time Setpoint Not Met (hr)']

        # Move 'Facility' as last row (Should always be in the index...)
        if 'Facility' in df_unmet.index:
            df_unmet = df_unmet.loc[[x for x
                                     in df_unmet.index
                                     if x != 'Facility'] + ['Facility']]

        return df_unmet

    def get_reporting_vars(self):
        """
        Queries 'ReportingDataDictionary' and returns a DataFrame

        Parameters
        -----------
        None

        Returns
        ---------
        df_vars: pd.DataFrame
            A DataFrame where each row is a reporting variable
        """

        with sqlite3.connect(self.sql_uri, uri=True) as con:
            query = '''
        SELECT KeyValue, Name, TimestepType, ReportingFrequency, Units, Type
            FROM ReportDataDictionary
            '''
            df_vars = pd.read_sql(query, con=con)

        return df_vars

    def get_hourly_variables(self, variables_list):
        """
        Queries Hourly variables which names are in variables_list

        eg: variables_list=['Zone Thermal Comfort CEN 15251 Adaptive Model Temperature']
        """

        query = '''
        SELECT EnvironmentPeriodIndex, Month, Day, Hour, Minute,
            ReportingFrequency, KeyValue, Name, Units,
            Value
        FROM ReportVariableWithTime
        '''

        cond = []

        cond.append(
            ("UPPER(Name) IN ({})".format(', '.join(
                map(repr, [name.upper() for name in variables_list]))))
        )

        cond.append('ReportingFrequency = "Hourly"')

        query += '  WHERE {}'.format('\n    AND '.join(cond))

        with sqlite3.connect(self.sql_uri, uri=True) as con:
            df = pd.read_sql(query, con=con)

        df_pivot = pd.pivot_table(df, values='Value',
                                  columns=['ReportingFrequency', 'KeyValue',
                                           'Name', 'Units'],
                                  index=['EnvironmentPeriodIndex',
                                         'Month', 'Day', 'Hour', 'Minute'])

        df_pivot = df_pivot.loc[3]  # Get the annual environment period index

        # We know it's hourly, so recreate a clear index
        (month, day, hour, minute) = df_pivot.index[0]
        start = datetime.datetime(2005, month, day)
        df_pivot.index = pd.date_range(
            start=start, periods=df_pivot.index.size, freq='H')
        df_pivot = df_pivot['Hourly']

        return df_pivot

    def get_timestep_variables(self, variables_list=None):
        """
        Queries 'Zone Timestep' variables which names are in variables_list (if supplied, otherwise all)

        eg: variables_list=['Zone Thermal Comfort CEN 15251 Adaptive Model Temperature']
        """

        query = '''
        SELECT EnvironmentPeriodIndex, Month, Day, Hour, Minute,
            ReportingFrequency, KeyValue, Name, Units,
            Value
        FROM ReportVariableWithTime
        '''

        cond = []

        if variables_list:
            cond.append(
                ("UPPER(Name) IN ({})".format(', '.join(
                    map(repr, [name.upper() for name in variables_list]))))
            )

        cond.append('ReportingFrequency = "Zone Timestep"')

        query += '  WHERE {}'.format('\n    AND '.join(cond))

        with sqlite3.connect(self.sql_uri, uri=True) as con:
            df = pd.read_sql(query, con=con)

        df_pivot = pd.pivot_table(df, values='Value',
                                  columns=['ReportingFrequency', 'KeyValue',
                                           'Name', 'Units'],
                                  index=['EnvironmentPeriodIndex',
                                         'Month', 'Day', 'Hour', 'Minute'])

        df_pivot = df_pivot.loc[3]  # Get the annual environment period index

        # We know it's Zone Timestep, with 15min timestep, so recreate a clear index
        (month, day, hour, minute) = df_pivot.index[0]
        start = datetime.datetime(2005, month, day)

        df_pivot.index = pd.date_range(
            start=start, periods=df_pivot.index.size, freq='15Min')
        df_pivot = df_pivot['Zone Timestep']

        return df_pivot


# In[ ]:


def modify_idf(idfname_init, epwfile, igu, run_n, df_step):
    """
    Modify the idf parameters, i.e. glazing, frame, and shadings, 
    according to the parameters defined in the dataframe,
    and returns the idf file

    Parameters
    ----------
    idfname_init: idf file to modify
    epwfile: weather data, .epw
    igu: name of the igu studied for energy simulation
    run_n: name/code for the energy simulation
    df_step: dataframe w/ a list of variables according to which are changed
        the idf parameters

    Returns
    -------
    idf_modified: a copy (saved as) of the initial idf
    """

    idf = IDF(idfname_init, epwfile)
    constructions = idf.idfobjects["CONSTRUCTION"]

    # Change the glazing and frame:
    for element in idf.idfobjects['FenestrationSurface:Detailed']:
        if element.Surface_Type == 'Window':

            # Replace the glazing:
            element.Construction_Name = igu
            if igu not in [
                construction.Name for construction in constructions
            ]:
                print('Wrong construction name!! See:', igu)

    # Replace the frame:
    for element in idf.idfobjects['WindowProperty:FrameAndDivider']:
        if igu in igu_low_perf:
            # if df_step.loc[df_step.index == run_n, 'thermal_curtain'][0] == 0:
            # element.Frame_Conductance = 5.5
            # element.Divider_Conductance = 5.5
            # else:
            element.Frame_Conductance = 1.56
            element.Divider_Conductance = 1.56

        if igu in igu_high_perf:
            # if df_step.loc[df_step.index == run_n, 'thermal_curtain'][0] == 0:
            # element.Frame_Conductance = 1.5
            # element.Divider_Conductance = 1.5
            # else:
            element.Frame_Conductance = 1
            element.Divider_Conductance = 1

    # Activate or not the shading control:
    for element in idf.idfobjects['WindowShadingControl']:
        element.Shading_Device_Material_Name = 'EnviroScreen_lightgrey_silver'
        element.Shading_Control_Type = 'OnIfHighZoneAirTempAndHighSolarOnWindow'
        element.Schedule_Name = 'Shading_On'
        element.Shading_Control_Is_Scheduled = 'Yes'
        element.Glare_Control_Is_Active = 'No'
        element.Type_of_Slat_Angle_Control_for_Blinds = 'FixedSlatAngle'

        if df_step.loc[df_step.index == run_n, 'int_shdg_device'][0] == 1:
            element.Setpoint = '26'
            element.Setpoint_2 = '100'
            element.Shading_Type = 'InteriorShade'

        elif df_step.loc[df_step.index == run_n, 'ext_shdg_device'][0] == 1:
            element.Setpoint = '22'
            element.Setpoint_2 = '100'
            element.Shading_Type = 'InteriorShade'
            element.Shading_Type = 'ExteriorShade'

        else:
            element.Schedule_Name = 'Shading_OFF'

    # Define heating and cooling setpoint:
    for element in idf.idfobjects['ThermostatSetpoint:DualSetpoint']:
        if ('Core' in element.Name
                or 'Perimeter' in element.Name):
            element.Heating_Setpoint_Temperature_Schedule_Name = (
                df_step.loc[df_step.index == run_n, 'heating_setpoint'][0])
            element.Cooling_Setpoint_Temperature_Schedule_Name = (
                df_step.loc[df_step.index == run_n, 'cooling_setpoint'][0])
    
    # Subdirectory with idf:    
    idf_dir = os.path.join(IDF_DIR)

    idf.saveas(idf_dir+"\BEM_"+str(run_n)+".idf")
    idf_modified = IDF(idf_dir+"\BEM_"+str(run_n)+".idf", epwfile)

    print("Saved: BEM_"+str(run_n))

    return idf_modified


# In[ ]:


def simulation_postprocess(run_n, path, 
                           ls_df_steps, ls_unmet, df_end_use_allsteps):
    """
    Run a simulation from an idf, extract results in df or ls:
    > total energy end use in df_end_use (electricity + natural gas)
    > unmethours during occupation in ls_unmet
    > Energy consumption per enduse in df_end_use_allsteps

    Save also the hourly reporting variables per simulation run 
    in a csv file: df_h_run.csv.

    Parameters
    ----------
    run_n: name/code for the energy simulation
    df_step: df define for each step of the LCA where to save
        values for electricity and natural gas use.

    Returns
    -------
    df_step: electricity use in kWh, use of nat gas in MJ
    ls_unmet
    df_end_use_allsteps: values in GJ

    """
    
    for df in ls_df_steps:
        if run_n in df.index:
            df_step = df
    
    # Find the output data:
    eplus_sql = EPLusSQL(sql_path=path+'\eplusout.sql')

    # Get total(i.e. whole building) energy end use in a dataframe, in GJ:
    df_end_use = eplus_sql.get_annual_energy_by_fuel_and_enduse()
    if 'Water' in df_end_use.columns:
        df_end_use = df_end_use.drop('Water', axis=1)
    df_end_use = df_end_use.drop([
        'Exterior Lighting', 'Exterior Equipment', 'Generators',
        'Water Systems', 'Heat Recovery', 'Humidification', 'Refrigeration'])

    # Save total elec and nat gas use for LCA in df_step:
    # Sum of the electricity uses:
    elec_tot_gj = df_end_use[('Electricity', 'GJ')].sum()
    # Convert GJ to kWh:
    elec_tot_kwh = elec_tot_gj * 277.8

    # Use of natural gas:
    if 'Natural Gas' not in df_end_use.columns:
        df_end_use[('Natural Gas', 'GJ')] = 0

    # Use of natural gas:
    gas_tot_gj = df_end_use[('Natural Gas', 'GJ')].sum()
    # Convert GJ to MJ:
    gas_tot_mj = gas_tot_gj * 1000
    
    # Save values in df_step:
    # elec: kWh/m² of glazed façade
    df_step.loc[df_step.index == run_n, 'elec_use'] = (
        elec_tot_kwh / glazed_facade_area)
    # gas: MJ/m² of glazed façade
    df_step.loc[df_step.index == run_n, 'natural_gas'] = (
        gas_tot_mj / glazed_facade_area)

    # Append the list of unmet hours during occupied cooling/heating:
    df_unmet = eplus_sql.get_unmet_hours_table()
    val_toadd = {'Run': run_n,
                 'During cooling': df_unmet.loc[df_unmet.index == 'Facility',
                                                'During Occupied Cooling'][0],
                 'During heating': df_unmet.loc[df_unmet.index == 'Facility',
                                                'During Occupied Heating'][0]
                 }
    
    # Avoid duplicating values:
    if len(ls_unmet) > 0:
        for i in range(len(ls_unmet)):
            if ls_unmet[i]['Run'] == val_toadd['Run']:
                # Replace old value:
                ls_unmet[i] = val_toadd
    else:
        # And append if did not exist before:
        ls_unmet.append(val_toadd)

    # Save energy consumption per end use, GJ, whole building:
    df_end_use = df_end_use.stack(['FuelType'])
    df_end_use['Run name'] = run_n
    df_end_use = df_end_use.reset_index().pivot(
        index='EndUse', columns=['Run name', 'FuelType'], values='GJ')
    
    if not df_end_use_allsteps.empty:
        # Columns to avoid when merging, avoid duplicates and save new values:
        cols_to_use = df_end_use_allsteps.columns.difference(
            df_end_use.columns)
        # Merge by columns_to_use:
        df_end_use_allsteps = pd.merge(df_end_use,
                                       df_end_use_allsteps[cols_to_use],
                                       on="EndUse")
    else:
        df_end_use_allsteps = df_end_use

    df_end_use_allsteps = df_end_use_allsteps.reindex(
        sorted(df_end_use_allsteps.columns), axis=1
    )

    # Hourly reporting variables
    # Define an empty DataFrame to save the hourly reporting variables:
    df_h_run = pd.DataFrame()

    ls_vars = [
        'Zone Windows Total Heat Gain Energy',
        'Zone Windows Total Heat Loss Energy',
        'Surface Shading Device Is On Time Fraction',
        'Zone Operative Temperature',
        'Zone Thermal Comfort CEN 15251 Adaptive Model Temperature',
        'Zone Thermal Comfort ASHRAE 55 Adaptive Model Temperature',
        'Chiller Electricity Energy', 'Boiler Heating Energy'
    ]

    df_h_run = eplus_sql.get_hourly_variables(variables_list=ls_vars)

    for col in df_h_run.columns:
        if "BASEMENT" in col[0]:
            df_h_run = df_h_run.drop(col, axis=1)

    # Save df_h_run to csv:
    df_h_run.to_csv('outputs\steps_dir\df_h_run_'+str(run_n[:3])+'.csv',
                    index=True)

    return run_n, df_step, ls_unmet, df_end_use_allsteps


# In[ ]:


def save_results_csv(run_n, df_step, ls_unmet, df_end_use_allsteps):
    """
    Save the DataFrames and Lists where the results are to avoid 
    reruning the time consuming energy simulation

    Parameters
    ----------
    run_n: simulation
    df_step: df define for each step of the LCA where to save
        values for electricity and natural gas use.
    ls_unmet: list for unmet hours during occupation
    df_end_use_allsteps: dataframe with end uses 
        per type of energy per simulation run

    Returns
    -------
    None

    """
    
    n = [ord(run_n[0]) - 96]

    # Save df_step to csv:
    df_step.to_csv('outputs\steps_dir\df_step'+str(n[0])+'.csv', index=True)
    
    # Save ls_unmet to csv:
    df_ls_unmet = pd.DataFrame(ls_unmet)
    df_ls_unmet.to_csv('outputs\steps_dir\df_ls_unmet.csv', index=False)
    
    # Save df_end_use_allsteps to csv:
    df_end_use_allsteps.stack([0, 1]).to_csv(
        'outputs\steps_dir\df_end_use_allsteps.csv', index=True)

    return


# In[ ]:


def run_single_simulation(args):
    """
    Gets a tuple of arguments, 
        eg: (file1.idf, weather.epw, run_n)
    """
    idf_path = os.path.relpath(args[0], ORIGIN_DIR)
    epw_path = os.path.relpath(args[1], ORIGIN_DIR)

    idf, idf_ext = os.path.splitext(idf_path)
    epw, epw_ext = os.path.splitext(epw_path)
    
    out_dir = os.path.relpath(os.path.join(OUT_DIR_EPlus, args[2]), ORIGIN_DIR)

    cmd = f'energyplus -w {epw_path} -d {out_dir} {idf_path}'
    res = subprocess.run(cmd, capture_output=True)
    
    if res.returncode != 0:
        print("Simulation failed for {idf_path} / {epw_path}")
        print(res.stdout.decode())
        print(res.stderr.decode())
        print("\n\n")
        
    return args[2], out_dir

