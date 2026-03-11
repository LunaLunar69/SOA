import pandas as pd

pd.set_option('display.max_rows', 20000)

def domar_al_monstruo():
    ruta_parquet = "HighQualityDataset.parquet"
    
    print("Leeyendo parquet...")
    
    df = pd.read_parquet(ruta_parquet, engine='pyarrow')
    
    print(f"Se cargaron {len(df):,} filas.")
    
    print("\nAsomándonos a los datos:")
    print(df.iloc[0:6000]) 
    
    print("\nInformación de las columnas:")
    print(df.info())

if __name__ == "__main__":
    domar_al_monstruo()