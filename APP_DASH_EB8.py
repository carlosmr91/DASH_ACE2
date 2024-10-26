# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:44:54 2024

@author: juan.melendez
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import plotly.express as px

# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
def configure_page():
    st.set_page_config(page_title="MONITOREO PRODUCCIÓN")
    st.markdown("<h1 style='text-align: center; color: black;'>HISTÓRICO DE PRODUCCIÓN ACE</h1>", 
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: green;'>TABLERO DE PRODUCCIÓN ALOCADA A AGOSTO 2024</h3>", 
                unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='color: gray;'>⚙️ CONTROLADORES</h2>", 
                        unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center;'>---------------------------------</p>", 
                        unsafe_allow_html=True)

# CARGA DE ARCHIVO
def load_data():
    uploaded_file = st.sidebar.file_uploader("📂 Cargar archivo (formato CSV UTF-8)", type=["csv", "CSV", "TXT", "txt"])
    if uploaded_file:
        return pd.read_csv(uploaded_file, sep=",")
    st.toast("ARCHIVO NO CARGADO ❗❗")
    st.stop()

# PROCESAMIENTO DE DATOS
def process_data(df):
    df["Fecha"] = pd.to_datetime(df["Fecha"], format='%d/%m/%Y %H:%M')
    columns_of_interest = ["Pozo_Oficial", "CAMPO", "MAESTRA BATERIA", "MAESTRA CAMPANA", "Fecha", "NumeroMeses",
                            "BrutoAcumulado Mbbl","AceiteAcumulado Mbbl", "AguaAcumulada Mbbl", "GasAcumulado MMpc",
                            "BrutoDiario bpd","AceiteDiario bpd", "AguaDiaria bpd", "GasDiario pcd",  "RPM", "YACIMIENTO","RGA pcb"]
    ofm_df = df[columns_of_interest]
    ofm_df["RGL pcb"] = ofm_df["GasDiario pcd"] / ofm_df["BrutoDiario bpd"]
    
    # Agrupación y fusión
    max_values = ofm_df.groupby("Pozo_Oficial")[["AceiteAcumulado Mbbl", "GasAcumulado MMpc", "AguaAcumulada Mbbl", "NumeroMeses","RGA pcb","RGL pcb"]].max().reset_index()
    max_values = pd.merge(max_values, ofm_df[["Pozo_Oficial","MAESTRA CAMPANA"]].drop_duplicates(), on="Pozo_Oficial", how="left")
    
    return ofm_df, max_values

def parametros_iniciales(df, selected_pozos):
    # Conversión de la columna Fecha a tipo datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"], format='%d/%m/%Y %H:%M')

    # Filtrar el DataFrame según los pozos seleccionados
    df_filtered = df[df["Pozo_Oficial"].isin(selected_pozos)]
    
    # Definir las columnas de interés
    columns_of_interest = ["Pozo_Oficial", "MAESTRA CAMPANA", "Fecha", "NumeroMeses",
                           "BrutoAcumulado Mbbl", "AceiteAcumulado Mbbl", "AguaAcumulada Mbbl", "GasAcumulado MMpc",
                           "BrutoDiario bpd", "AceiteDiario bpd", "AguaDiaria bpd", "GasDiario pcd", "RPM", 
                           "YACIMIENTO", "RGA pcb"]

    # Filtrar el DataFrame solo con las columnas de interés
    ofm_df = df_filtered[columns_of_interest]

    # Filtrar los datos donde 'NumeroMeses' es igual a 1
    iniciales_df = ofm_df[ofm_df["NumeroMeses"] == 1]
    tabla_iniciales = iniciales_df[["Pozo_Oficial", "MAESTRA CAMPANA", "Fecha", 
                                    "BrutoDiario bpd", "AceiteDiario bpd", "AguaDiaria bpd", 
                                    "GasDiario pcd", "RGA pcb"]]  
    
    tabla_iniciales.rename(columns={
        "Pozo_Oficial": "POZO",
        "MAESTRA CAMPANA": "CAMPAÑA",
        "Fecha": "FECHA INICIO PRODUCCIÓN",
        "BrutoDiario bpd": "QBruto Inicial (bpd)",
        "AceiteDiario bpd": "QNeta Inicial (bpd)",
        "AguaDiaria bpd": "QAgua Inicial (bpd)",
        "GasDiario pcd": "QGas Inicial (pcd)",
        "RGA pcb": "RGA Inicial (pcb)",
    }, inplace=True)
    
    maximos_df = ofm_df.groupby("Pozo_Oficial")[["NumeroMeses","AceiteDiario bpd", "AguaDiaria bpd", 
                                    "GasDiario pcd", "RGA pcb","BrutoAcumulado Mbbl","AceiteAcumulado Mbbl",
                                    "AguaAcumulada Mbbl", "GasAcumulado MMpc"]].max().reset_index()
    tabla_maximos = pd.merge(maximos_df, ofm_df[["Pozo_Oficial"]].drop_duplicates(), 
                             on="Pozo_Oficial", how="left")
    
    # Renombrar las columnas del DataFrame
    tabla_maximos.rename(columns={
        "Pozo_Oficial": "POZO",
        "AceiteDiario bpd": "QNeta Máximo (bpd)",
        "AguaDiaria bpd": "QAgua Máximo (bpd)",
        "GasDiario pcd": "QGas Máximo (pcd)",
        "RGA pcb": "RGA Máxima (pcb)",
        "NumeroMeses": "MESES ACTIVO",
        "BrutoAcumulado Mbbl": "Nb (Mbbl)",
        "AceiteAcumulado Mbbl": "Np (Mbbl)",
        "AguaAcumulada Mbbl": "Wp (Mbbl)",
        "GasAcumulado MMpc": "Gp (MMpc)"
    }, inplace=True)

    recent_dates = ofm_df.groupby('Pozo_Oficial')['Fecha'].max().reset_index()
    recent_values_df = pd.merge(recent_dates, ofm_df, on=['Pozo_Oficial', 'Fecha'], how='left')
    final_df = recent_values_df[["Pozo_Oficial", "Fecha", "RPM", "BrutoDiario bpd", "AceiteDiario bpd", 
                                 "AguaDiaria bpd", "GasDiario pcd", "RGA pcb"]]

    final_df.rename(columns={
        "Pozo_Oficial": "POZO",
        "Fecha": "FECHA UCV",
        "BrutoDiario bpd": "QBruto UCV (bpd)",
        "AceiteDiario bpd": "QNeta UCV (bpd)",
        "AguaDiaria bpd": "QAgua UCV (bpd)",
        "GasDiario pcd": "QGas UCV (pcd)",
        "RPM": "RPM UCV",
        "RGA pcb": "RGA UCV (pcb)",
    }, inplace=True)  
    
    # Función auxiliar para crear tablas en Plotly con 2 decimales
    def crear_tabla(df, ancho, alto):
        # Formatear los valores numéricos con 2 decimales
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if df_formatted[col].dtype in ['float64', 'int64']:
                df_formatted[col] = df_formatted[col].map('{:.2f}'.format)

        fig = go.Figure(data=[go.Table(
          header=dict(
            values=list(df_formatted.columns),
            line_color='darkslategray',
            fill_color='grey',
            align=['left', 'center'],
            font=dict(color='white', size=13)
          ),
          cells=dict(
            values=[df_formatted[col] for col in df_formatted.columns],
            line_color='darkslategray',
            fill_color = [['lightgrey', 'white']*len(df_formatted)],
            align = ['left', 'center'],
            font = dict(color = 'darkslategray', size = 13)
            ))
        ])
        fig.update_layout(
            width=ancho,  # Anchura de la tabla en píxeles
            height=alto   # Altura de la tabla en píxeles
        )
        return fig

    # Crear las tablas
    tabla1_fig = crear_tabla(tabla_iniciales, 850, 500)
    tabla2_fig = crear_tabla(final_df, 900, 500)
    tabla3_fig = crear_tabla(tabla_maximos, 900, 500)
    
    # Devolver las tablas y las figuras
    return tabla_iniciales, final_df, tabla_maximos, tabla1_fig, tabla2_fig, tabla3_fig


    #st.write(df_filtered["Pozo_Oficial"].unique())
    #st.write(ofm_df["Pozo_Oficial"].unique())
    #st.write(tabla_maximos["Pozo_Oficial"].unique())

def contador_diario(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    # Filtrar datos mayores a cero
    df_filtered = df[df[y_col] > 0]
    
    # Contar el número de valores mayores a cero por fecha
    df_count = df_filtered.groupby('Fecha').size().reset_index(name='Count')
    
    # Crear gráfico de líneas para el conteo
    fig = px.line(df_count, x="Fecha", y="Count", title=titulo,
                  markers=False, line_shape="linear", line_dash_sequence=["solid"],
                  color_discrete_sequence=["black"])
    
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=600,
        height=280,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=140, t=40, b=0),
        yaxis=dict(title=yaxis_title, side='left', 
                    showgrid=True, gridcolor='LightGray', gridwidth=1, 
                    zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y        
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                    gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje x
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje x
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )
    
    fig.update_traces(line=dict(width=4))
    return fig


# FUNCION PARA GRAFICOS DE PRODUCCIÓN DIARIA
def plots_diaria(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    # Calcular la suma de la columna seleccionada
    df_sum = df.groupby('Fecha').agg({y_col: 'sum'}).reset_index()
    
    # Calcular el rango máximo del eje y
    max_y = max(df[y_col].max(), df_sum[y_col].max()) * 1.2
    
    # Crear gráfico de líneas para cada pozo
    fig = px.line(df, x="Fecha", y=y_col, color=color_col,
                  title=titulo, line_shape="linear")
    
    # Añadir la suma total de los pozos seleccionados como una nueva traza
    fig.add_trace(go.Scatter(x=df_sum['Fecha'], y=df_sum[y_col],
                             mode=mode,
                             name='Suma Total',
                             marker=dict(color='black', symbol='cross', size=marker_size),
                             yaxis='y1'))
    
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=600,
        height=280,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_y], title=yaxis_title, side='left', 
                   showgrid=True, gridcolor='LightGray', gridwidth=1, 
                   zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
                   title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                   tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                   gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1,
                   title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje x
                   tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje x
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )
    
    fig.update_traces(line=dict(width=4))
    
    return fig

# FUNCION PARA GRAFICOS DE PRODUCCIÓN DIARIA NORMALIZADA
def plots_diaria_norm(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    # Calcular la suma de la columna seleccionada
    df_sum = df.groupby('NumeroMeses').agg({y_col: 'sum'}).reset_index()
    
    # Calcular el rango máximo del eje y
    max_y = max(df[y_col].max(), df_sum[y_col].max()) * 1.2
    
    # Crear gráfico de líneas para cada pozo
    fig = px.line(df, x="NumeroMeses", y=y_col, color=color_col,
                  title=titulo, line_shape="linear")
    
    # Añadir la suma total de los pozos seleccionados como una nueva traza
    fig.add_trace(go.Scatter(x=df_sum['NumeroMeses'], y=df_sum[y_col],
                             mode=mode,
                             name='Suma Total',
                             marker=dict(color='black', symbol='cross', size=marker_size),
                             yaxis='y1'))
    
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=700,
        height=280,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_y], title=yaxis_title, side='left', 
                   showgrid=True, gridcolor='LightGray', gridwidth=1, 
                   zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
                   title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                   tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        xaxis=dict(title="Meses", showgrid=True, gridcolor='LightGray', 
                   gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1,
                   title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje x
                   tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje x
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )
    
    fig.update_traces(line=dict(width=4))
    return fig

def plots_acumulada(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    
    # Calcular el rango máximo del eje y
    max_y = df[y_col].max()* 1.2
    
    # Crear gráfico de líneas para cada pozo
    fig = px.line(df, x="Fecha", y=y_col, color=color_col,
                  title=titulo, line_shape="linear")
        
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=600,
        height=280,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_y], title=yaxis_title, side='left', 
                    showgrid=True, gridcolor='LightGray', gridwidth=1, 
                    zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                    gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )
    
    fig.update_traces(line=dict(width=4))
    return fig



def plots_acumuladaTOTAL(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    # Verificar si y_col existe en el DataFrame
    if y_col not in df.columns:
        raise KeyError(f"La columna '{y_col}' no existe en el DataFrame")

    # Obtener la fecha máxima del DataFrame
    max_date = df['Fecha'].max()
    
    # Crear un DataFrame extendido con fechas que van desde la mínima hasta la fecha más reciente
    extended_dates = pd.date_range(start=df['Fecha'].min(), end=max_date, freq='MS')
    extended_df = pd.DataFrame({'Fecha': extended_dates})

    # Inicializar una figura
    fig = go.Figure()

    # Inicializar un DataFrame para la suma acumulada
    df_sum = pd.DataFrame({'Fecha': extended_dates})
    df_sum[y_col] = 0

    # Iterar sobre cada pozo único
    for pozo in df[color_col].unique():
        # Filtrar datos para el pozo actual
        df_pozo = df[df[color_col] == pozo].copy()

        # Verificar si el pozo tiene datos y fechas válidas
        if df_pozo['Fecha'].notna().any() and pd.notna(df_pozo['Fecha'].min()):
            # Extender el DataFrame del pozo actual hasta la fecha más reciente
            extended_df_pozo = pd.merge(extended_df, df_pozo[['Fecha', y_col]], on='Fecha', how='left')

            # Aplicar carry forward: rellenar valores faltantes con el último valor conocido
            extended_df_pozo[y_col].fillna(method='ffill', inplace=True)

            # Si quedan valores NaN después de fillna, rellenarlos con 0 o el valor deseado
            extended_df_pozo[y_col].fillna(0, inplace=True)

            # Añadir la traza al gráfico como línea continua
            fig.add_trace(
                go.Scatter(
                    x=extended_df_pozo['Fecha'],
                    y=extended_df_pozo[y_col],
                    hoverinfo='x+y',
                    mode='lines',
                    name=pozo
                )
            )

            # Sumar los valores al DataFrame de suma acumulada
            df_sum[y_col] += extended_df_pozo[y_col]

    # Añadir la traza de la suma acumulada
    fig.add_trace(
        go.Scatter(
            x=df_sum['Fecha'],
            y=df_sum[y_col],
            hoverinfo='x+y',
            mode='lines',  # Cambiado de 'cross' a 'lines'
            name='Suma Acumulada',
            line=dict(color='black', width=2)
        )
    )
    
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=600,
        height=280,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(title=yaxis_title, side='left', 
                    showgrid=True, gridcolor='LightGray', gridwidth=1, 
                    zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                    gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y,
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )

    return fig


# FUNCION PARA GRAFICOS DE PRODUCCIÓN ACUMULADA
def plots_acumulada_norm(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):
    
    # Calcular el rango máximo del eje y
    max_y = df[y_col].max()* 1.2
    
    # Crear gráfico de líneas para cada pozo
    fig = px.line(df, x="NumeroMeses", y=y_col, color=color_col,
                  title=titulo, line_shape="linear")
        
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=700,
        height=280,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_y], title=yaxis_title, side='left', 
                    showgrid=True, gridcolor='LightGray', gridwidth=1, 
                    zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        xaxis=dict(title="Meses", showgrid=True, gridcolor='LightGray', 
                    gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )
    
    fig.update_traces(line=dict(width=4))
    return fig



def plots_acumulada_apilada(df, y_col, titulo, color_col, color, mode, yaxis_title, paper_bgcolor, marker_size=4):

    # Verificar si y_col existe en el DataFrame
    if y_col not in df.columns:
        raise KeyError(f"La columna '{y_col}' no existe en el DataFrame")

    # Obtener la fecha máxima del DataFrame
    max_date = df['Fecha'].max()
    # Crear un DataFrame extendido con fechas que van desde la mínima hasta la fecha más reciente
    extended_dates = pd.date_range(start=df['Fecha'].min(), end=max_date, freq='MS')
    extended_df = pd.DataFrame({'Fecha': extended_dates})

    # Inicializar una figura
    fig = go.Figure()

    # Iterar sobre cada pozo único
    for pozo in df[color_col].unique():
        # Filtrar datos para el pozo actual
        df_pozo = df[df[color_col] == pozo].copy()

        # Verificar si el pozo tiene datos y fechas válidas
        if df_pozo['Fecha'].notna().any() and pd.notna(df_pozo['Fecha'].min()):
            # Extender el DataFrame del pozo actual hasta la fecha más reciente
            extended_df_pozo = pd.merge(extended_df, df_pozo[['Fecha', y_col]], on='Fecha', how='left')

            # Aplicar carry forward: rellenar valores faltantes con el último valor conocido
            extended_df_pozo[y_col].fillna(method='ffill', inplace=True)

            # Si quedan valores NaN después de fillna, rellenarlos con 0 o el valor deseado
            extended_df_pozo[y_col].fillna(0, inplace=True)

            # Añadir la traza al gráfico
            fig.add_trace(
                go.Scatter(
                    x=extended_df_pozo['Fecha'],
                    y=extended_df_pozo[y_col],
                    hoverinfo='x+y',
                    mode='none',
                    name=pozo,
                    fill='tonexty',
                    stackgroup='one'
                )
            )

    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=600,
        height=280,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(title=yaxis_title, side='left', 
                    showgrid=True, gridcolor='LightGray', gridwidth=1, 
                    zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        xaxis=dict(title="Fecha", showgrid=True, gridcolor='LightGray', 
                    gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y,
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )

    return fig

# FUNCION PARA GRAFICOS DE PRODUCCIÓN ACUMULADA
def relacion_diaria_acum(df, x_col, y_col, titulo, color_col, color, mode, xaxis_title, yaxis_title, paper_bgcolor, marker_size=4):
    
    # Calcular el rango máximo del eje y
    max_y = df[y_col].max()* 1.2
    
    # Crear gráfico de líneas para cada pozo
    fig = px.line(df, x=x_col, y=y_col, color=color_col,
                  title=titulo, line_shape="linear")
        
    # DISEÑO DE GRÁFICO
    fig.update_layout(
        title=titulo,
        width=700,
        height=280,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis=dict(range=[0, max_y], title=yaxis_title, side='left', 
                    showgrid=True, gridcolor='LightGray', gridwidth=1, 
                    zeroline=True, zerolinecolor='LightGray', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        xaxis=dict(title=xaxis_title, showgrid=True, gridcolor='LightGray', 
                    gridwidth=1, zeroline=True, zerolinecolor='Black', zerolinewidth=1,
                    title_font=dict(size=14, family='Arial Black'),  # Modificar la fuente del título del eje y
                    tickfont=dict(size=12, family='Arial Black')),   # Modificar la fuente de los ticks del eje y
        legend=dict(font=dict(size=12, family="Arial", color="black"))
    )
    
    fig.update_traces(line=dict(width=4))
    return fig

# ESTRUCTURA DE PÁGINA
def main():
    # Configura la página y carga los datos
    configure_page()
    df = load_data()
    
    with st.spinner("Procesando datos..."):
        # Procesa los datos
        ofm_df, ofm_max = process_data(df)
        
        # Crea las pestañas de la interfaz
        tabs = st.tabs(["PRODUCCIÓN DIARIA", " PRODUCCIÓN DIARIA NORMALIZADA","RELACIONES", "RESUMEN"])                
        
        # Pestaña "PRODUCCIÓN DIARIA"
        with tabs[0]:
            # Selección de yacimientos
            selected_yaci = st.sidebar.multiselect("SELECCIONA EL/LOS YACIMIENTO(S)", ofm_df["YACIMIENTO"].unique())
            if not selected_yaci:
                st.warning("Por favor selecciona al menos un yacimiento.")
                return  # Salir de la función si no se ha seleccionado un yacimiento
            df_yaci = ofm_df[ofm_df["YACIMIENTO"].isin(selected_yaci)]
            
            # Selección de pozos
            selected_pozos = st.sidebar.multiselect("SELECCIONA EL/LOS POZO(S)", df_yaci["Pozo_Oficial"].unique())
            if not selected_pozos:
                st.warning("Por favor selecciona al menos un pozo.")
                return  # Salir de la función si no se ha seleccionado un pozo

            c1, c2, c3 = st.columns(3)
            
            # Columna 1: Mostrar gráficos
            with c1:         
                if selected_pozos:
                    # Filtrar datos por pozos seleccionados
                    df_xpozoyaci = df_yaci[df_yaci["Pozo_Oficial"].isin(selected_pozos)]
                    
                    # Verificar y limpiar datos de la columna 'Fecha'
                    if df_xpozoyaci['Fecha'].notna().any():
                        df_xpozoyaci = df_xpozoyaci.dropna(subset=['Fecha']).sort_values(by='Fecha')
                        
                        # Crear gráficos usando funciones modularizadas
                        figBruto = plots_diaria(
                            df_xpozoyaci, 'BrutoDiario bpd', "PRODUCCIÓN BRUTA DIARIA (bpd)", "Pozo_Oficial", 
                            'black', 'markers', "Bruta (bpd)", "#ECECEC"
                        )
                        figNeta = plots_diaria(
                            df_xpozoyaci, 'AceiteDiario bpd', "PRODUCCIÓN NETA DIARIA (bpd)", "Pozo_Oficial", 
                            'black', 'markers', "Neta (bpd)", "#E5FDDF"
                        )
                        figAgua = plots_diaria(
                            df_xpozoyaci, 'AguaDiaria bpd', "PRODUCCIÓN DE AGUA DIARIA (bpd)", "Pozo_Oficial", 
                            'black', 'markers', "Agua (bpd)", "#DFF9FD"
                        )
                        figGas = plots_diaria(
                            df_xpozoyaci, 'GasDiario pcd', "PRODUCCIÓN DE GAS DIARIA (pcd)", "Pozo_Oficial", 
                            'black', 'markers', "Gas (pcd)", "#FEEDE8"
                        )

                        figContador = contador_diario(
                            df_xpozoyaci, 'AceiteDiario bpd', "TOTAL DE POZOS ACTIVOS", "Pozo_Oficial", 
                            'black', 'markers', "Pozos Activos", "#EFEFFF"
                        )
                        
                        # Mostrar los gráficos en la columna 1
                        for fig in [figBruto, figNeta, figAgua, figGas, figContador]:
                            c1.plotly_chart(fig)
                            
            with c2:                               
                        # Crear gráficos usando funciones modularizadas
                        figBrutoAC = plots_acumulada(
                            df_xpozoyaci, 'BrutoAcumulado Mbbl', "PRODUCCIÓN BRUTA ACUMULADA (Mbbl)", "Pozo_Oficial", 
                            'black', 'markers', "Bruta (Mbbl)", "#ECECEC"
                        )
                        figNetaAC = plots_acumulada(
                            df_xpozoyaci, 'AceiteAcumulado Mbbl', "PRODUCCIÓN NETA ACUMULADA (Mbbl)", "Pozo_Oficial", 
                            'black', 'markers', "Neta (Mbbl)", "#E5FDDF"
                        )
                        figAguaAC = plots_acumulada(
                            df_xpozoyaci, 'AguaAcumulada Mbbl', "PRODUCCIÓN DE AGUA ACUMULADA (Mbbl)", "Pozo_Oficial", 
                            'black', 'markers', "Agua (Mbbl)", "#DFF9FD"
                        )
                        figGasAC = plots_acumulada(
                            df_xpozoyaci, 'GasAcumulado MMpc', "PRODUCCIÓN DE GAS ACUMULADA (MMpc)", "Pozo_Oficial", 
                            'black', 'markers', "Gas (MMpc)", "#FEEDE8"
                        )
                        
                        figRPM = plots_diaria(
                            df_xpozoyaci, 'RPM', "REVOLUCIONES POR MINUTO", "Pozo_Oficial", 
                            'black', 'markers', "RPM", "#EFEFFF"
                        )                 
                        # Mostrar los gráficos en la columna 2
                        for fig in [figBrutoAC, figNetaAC, figAguaAC, figGasAC,figRPM]:
                            c2.plotly_chart(fig)                            
            
            with c3:
                        figBrutoACa = plots_acumulada_apilada(
                            df_xpozoyaci, 'BrutoAcumulado Mbbl', "PRODUCCIÓN BRUTA ACUMULADA (Mbbl) -Apilada", "Pozo_Oficial", 
                            'black', 'markers', "Bruta (Mbbl)", "#ECECEC"
                            )
                        figNetaACa = plots_acumulada_apilada(
                            df_xpozoyaci, 'AceiteAcumulado Mbbl', "PRODUCCIÓN NETA ACUMULADA (Mbbl) -Apilada", "Pozo_Oficial",
                            'black', 'markers', "Neta (Mbbl)", "#E5FDDF"
                            )
                        figAguaACa = plots_acumulada_apilada(
                            df_xpozoyaci, 'AguaAcumulada Mbbl', "PRODUCCIÓN DE AGUA ACUMULADA (Mbbl) -Apilada", "Pozo_Oficial",
                            'black', 'markers', "Agua (Mbbl)", "#DFF9FD"
                            )
                        figGasACa = plots_acumulada_apilada(
                            df_xpozoyaci, 'GasAcumulado MMpc', "PRODUCCIÓN DE GAS ACUMULADA (MMpc) -Apilada", "Pozo_Oficial", 
                            'black', 'markers', "Gas (MMpc)", "#FEEDE8"
                            )
                        figRGA = plots_diaria(
                            df_xpozoyaci, 'RGA pcb', "RGA (pcb)", "Pozo_Oficial", 
                            'black', 'markers', "Gas (pcd)", "#ECECEC"
                        )                       

                        
                        # Mostrar los gráficos en la columna 3
                        for fig in [figBrutoACa,figNetaACa,figAguaACa,figGasACa,figRGA]:
                            c3.plotly_chart(fig)
                            
        # Pestaña "PRODUCCION DIARIA NORMALIZADA"
        with tabs[1]:
            c1, c2 = st.columns(2)
            
            # Columna 1: Mostrar gráficos
            with c1:         
                if selected_pozos:
                    # # Filtrar datos por pozos seleccionados
                    # df_xpozoyaci = df_yaci[df_yaci["Pozo_Oficial"].isin(selected_pozos)]
                    
                    # # Verificar y limpiar datos de la columna 'NumeroMeses'
                    # if df_xpozoyaci['NumeroMeses'].notna().any():
                    #     df_xpozoyaci = df_xpozoyaci.dropna(subset=['NumeroMeses']).sort_values(by='NumeroMeses')
                        
                        # Crear gráficos usando funciones modularizadas
                        figBruto_norm = plots_diaria_norm(
                            df_xpozoyaci, 'BrutoDiario bpd', "PRODUCCIÓN BRUTA DIARIA (bpd) -Normalizada", "Pozo_Oficial", 
                            'black', 'markers', "Bruta (bpd)", "#ECECEC"
                        )
                        figNeta_norm = plots_diaria_norm(
                            df_xpozoyaci, 'AceiteDiario bpd', "PRODUCCIÓN NETA DIARIA (bpd) -Normalizada", "Pozo_Oficial", 
                            'black', 'markers', "Neta (bpd)", "#E5FDDF"
                        )
                        figAgua_norm = plots_diaria_norm(
                            df_xpozoyaci, 'AguaDiaria bpd', "PRODUCCIÓN DE AGUA DIARIA (bpd) -Normalizada", "Pozo_Oficial", 
                            'black', 'markers', "Agua (bpd)", "#DFF9FD"
                        )
                        figGas_norm = plots_diaria_norm(
                            df_xpozoyaci, 'GasDiario pcd', "PRODUCCIÓN DE GAS DIARIA (pcd) -Normalizada", "Pozo_Oficial", 
                            'black', 'markers', "Gas (pcd)", "#FEEDE8"
                        )
                        figRPM_norm = plots_diaria_norm(
                            df_xpozoyaci, 'RPM', "REVOLUCIONES POR MINUTO", "Pozo_Oficial", 
                            'black', 'markers', "RPM", "#EFEFFF"
                        )
                          
                        # Mostrar los gráficos en la columna 1
                        for fig in [figBruto_norm, figNeta_norm, figAgua_norm, figGas_norm, figRPM_norm]:
                            c1.plotly_chart(fig)
            with c2:                               
                        # Crear gráficos usando funciones modularizadas
                        figBrutoAC_norm = plots_acumulada_norm(
                            df_xpozoyaci, 'BrutoAcumulado Mbbl', "PRODUCCIÓN BRUTA ACUMULADA (Mbbl) -Normalizada", "Pozo_Oficial", 
                            'black', 'markers', "Bruta (Mbbl)", "#ECECEC"
                        )
                        figNetaAC_norm = plots_acumulada_norm(
                            df_xpozoyaci, 'AceiteAcumulado Mbbl', "PRODUCCIÓN NETA ACUMULADA (Mbbl) -Normalizada", "Pozo_Oficial", 
                            'black', 'markers', "Neta (Mbbl)", "#E5FDDF"
                        )
                        figAguaAC_norm = plots_acumulada_norm(
                            df_xpozoyaci, 'AguaAcumulada Mbbl', "PRODUCCIÓN DE AGUA ACUMULADA (Mbbl) -Normalizada", "Pozo_Oficial", 
                            'black', 'markers', "Agua (Mbbl)", "#DFF9FD"
                        )
                        figGasAC_norm = plots_acumulada_norm(
                            df_xpozoyaci, 'GasAcumulado MMpc', "PRODUCCIÓN DE GAS ACUMULADA (MMpc) -Normalizada", "Pozo_Oficial", 
                            'black', 'markers', "Gas (MMpc)", "#FEEDE8"
                        )

                                         
                        # Mostrar los gráficos en la columna 2
                        for fig in [figBrutoAC_norm, figNetaAC_norm, figAguaAC_norm, figGasAC_norm]:
                            c2.plotly_chart(fig)  


        with tabs[2]:
            c1, c2 = st.columns(2)
            
            # Columna 1: Mostrar gráficos
            with c1:         
                if selected_pozos:
                           
                        # Crear gráficos usando funciones modularizadas
                        figQB_BrutoAC = relacion_diaria_acum(
                            df_xpozoyaci,'BrutoAcumulado Mbbl', 'BrutoDiario bpd', "RELACIÓN BRUTA ACUMULADA (Mbbl) - BRUTA DIARIA (bpd)", "Pozo_Oficial", 
                            'black', 'markers',"Bruta Acumulada (Mbbl)", "Bruta (bpd)", "#ECECEC"
                        )
                        figQN_NetaAC = relacion_diaria_acum(
                            df_xpozoyaci, 'AceiteAcumulado Mbbl','AceiteDiario bpd', "RELACIÓN NETA ACUMULADA (Mbbl) - NETA DIARIA (bpd)", "Pozo_Oficial", 
                            'black', 'markers',"Neta Acumulada (Mbbl)", "Neta (bpd)", "#E5FDDF"
                        )
                        figQW_AguaAC = relacion_diaria_acum(
                            df_xpozoyaci, 'AguaAcumulada Mbbl','AguaDiaria bpd', "RELACIÓN AGUA ACUMULADA (Mbbl) - AGUA DIARIA (bpd)", "Pozo_Oficial", 
                            'black', 'markers', "Agua Acumulada (Mbbl)","Agua (bpd)", "#DFF9FD"
                        )
                        figGP_GasAC = relacion_diaria_acum(
                            df_xpozoyaci, 'GasAcumulado MMpc', 'GasDiario pcd', "RELACIÓN GAS ACUMULADO (MMpc)- GAS DIARIO (pcd)", "Pozo_Oficial", 
                            'black', 'markers',"Gas Acumulado (MMpc)", "Gas (pcd)", "#FEEDE8"
                        )
                        # Mostrar los gráficos en la columna 2
                        for fig in [figQB_BrutoAC,figQN_NetaAC,figQW_AguaAC,figGP_GasAC ]:
                            c1.plotly_chart(fig)  

            # with c2:  
            #             figRGA_BrutoAC = relacion_diaria_acum(
            #                 df_xpozoyaci,'BrutoAcumulado Mbbl', 'RGA pcb', "RELACIÓN BRUTA ACUMULADA (Mbbl) - RGA (pcb)", "Pozo_Oficial", 
            #                 'black', 'markers',"Bruta Acumulada (Mbbl)", "RGA pcb (pcb)", "#ECECEC"
            #                 )
            #             figRGA_BrutoAC.update_yaxes(type="log", range=[0, 5], exponentformat="power", tickformat="power")
                        
            #             figRGA_NetaAC = relacion_diaria_acum(
            #                 df_xpozoyaci,'AceiteAcumulado Mbbl', 'RGA pcb', "RELACIÓN NETA ACUMULADA (Mbbl) - RGA (pcb)", "Pozo_Oficial", 
            #                 'black', 'markers',"Neta Acumulada (Mbbl)", "RGA pcb (pcb)", "#E5FDDF"
            #                 )
            #             figRGA_NetaAC.update_yaxes(type="log", range=[0, 5], exponentformat="power", tickformat="power")
                        
            #             figRGA_AguaAC = relacion_diaria_acum(
            #                 df_xpozoyaci,'AguaAcumulada Mbbl', 'RGA pcb', "RELACIÓN AGUA ACUMULADA (Mbbl) - RGA (pcb)", "Pozo_Oficial", 
            #                 'black', 'markers',"Agua Acumulada (Mbbl)", "RGA pcb (pcb)", "#DFF9FD"
            #                 )
            #             figRGA_AguaAC.update_yaxes(type="log", range=[0, 5], exponentformat="power", tickformat="power")
                        
            #             figRGA_GasAC = relacion_diaria_acum(
            #                 df_xpozoyaci,'GasAcumulado MMpc', 'RGA pcb', "RELACIÓN GAS ACUMULADO (MMpc) - RGA (pcb)", "Pozo_Oficial", 
            #                 'black', 'markers',"Gas Acumulado (MMpc)", "RGA pcb (pcb)", "#FEEDE8"
            #                 )
            #             figRGA_GasAC.update_yaxes(type="log", range=[0, 5], exponentformat="power", tickformat="power")
                                       
            #             # Mostrar los gráficos en la columna 2
            #             for fig in [ figRGA_BrutoAC,figRGA_NetaAC, figRGA_AguaAC,figRGA_GasAC ]:
            #                c2.plotly_chart(fig)       
            with c2:  
                        figRGL_BrutoAC = relacion_diaria_acum(
                            df_xpozoyaci,'BrutoAcumulado Mbbl', 'RGL pcb', "RELACIÓN BRUTA ACUMULADA (Mbbl) - RGL (pcb)", "Pozo_Oficial", 
                            'black', 'markers',"Bruta Acumulada (Mbbl)", "RGL pcb (pcb)", "#ECECEC"
                            )
                        figRGL_BrutoAC.update_yaxes(type="log", range=[0, 5], exponentformat="power", tickformat="power")
                        
                        figRGL_NetaAC = relacion_diaria_acum(
                            df_xpozoyaci,'AceiteAcumulado Mbbl', 'RGL pcb', "RELACIÓN NETA ACUMULADA (Mbbl) - RGL (pcb)", "Pozo_Oficial", 
                            'black', 'markers',"Neta Acumulada (Mbbl)", "RGL pcb (pcb)", "#E5FDDF"
                            )
                        figRGL_NetaAC.update_yaxes(type="log", range=[0, 5], exponentformat="power", tickformat="power")
                        
                        figRGL_AguaAC = relacion_diaria_acum(
                            df_xpozoyaci,'AguaAcumulada Mbbl', 'RGL pcb', "RELACIÓN AGUA ACUMULADA (Mbbl) - RGL (pcb)", "Pozo_Oficial", 
                            'black', 'markers',"Agua Acumulada (Mbbl)", "RGL pcb (pcb)", "#DFF9FD"
                            )
                        figRGL_AguaAC.update_yaxes(type="log", range=[0, 5], exponentformat="power", tickformat="power")
                        
                        figRGL_GasAC = relacion_diaria_acum(
                            df_xpozoyaci,'GasAcumulado MMpc', 'RGL pcb', "RELACIÓN GAS ACUMULADO (MMpc) - RGL (pcb)", "Pozo_Oficial", 
                            'black', 'markers',"Gas Acumulado (MMpc)", "RGL pcb (pcb)", "#FEEDE8"
                            )
                        figRGL_GasAC.update_yaxes(type="log", range=[0, 5], exponentformat="power", tickformat="power")
                                       
                        # Mostrar los gráficos en la columna 2
                        for fig in [ figRGL_BrutoAC,figRGL_NetaAC, figRGL_AguaAC,figRGL_GasAC ]:
                           c2.plotly_chart(fig)                                                                           


        with tabs[3]:
                c1, c2 = st.columns(2)
            
                if selected_pozos:
                    # Obtener las tablas y las figuras
                    tabla_iniciales, final_df, tabla_maximos, tabla1_fig, tabla2_fig, tabla3_fig = parametros_iniciales(df, selected_pozos)
                    
                    # Botón y gráfico para la tabla de iniciales
                    c1.download_button(
                        label="Descargar tabla Gastos Iniciales",
                        data=tabla_iniciales.to_csv(index=False),
                        file_name='tabla_iniciales.csv',
                        mime='text/csv'
                    )
                    c1.plotly_chart(tabla1_fig)
                    
                    # Botón y gráfico para la tabla final
                    c1.download_button(
                        label="Descargar tabla Gastos Actuales",
                        data=final_df.to_csv(index=False),
                        file_name='tabla_final.csv',
                        mime='text/csv'
                    )
                    c1.plotly_chart(tabla2_fig)
                    
                    # Botón y gráfico para la tabla de máximos
                    c2.download_button(
                        label="Descargar tabla Gastos Máximos",
                        data=tabla_maximos.to_csv(index=False),
                        file_name='tabla_maximos.csv',
                        mime='text/csv'
                    )
                    c2.plotly_chart(tabla3_fig)

   
if __name__ == "__main__":
    main()
    
