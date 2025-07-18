#!/usr/bin/env python3
import pandas as pd

def summarize(csv_path, examples_per_class=3):
    # Carga el CSV
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Total de oraciones
    total = len(df)
    print(f"Total de oraciones: {total}\n")
    
    # Añadir columna de longitud
    df['Length'] = df['Oración'].str.split().apply(len)
    
    # Agrupar por clasificación
    grp = df.groupby('Clasificación')
    
    summary_rows = []
    for cls, group in grp:
        count = len(group)
        pct = count / total * 100
        avg_len = group['Length'].mean()
        
        # Seleccionar ejemplos
        examples = group['Oración'].head(examples_per_class).tolist()
        
        summary_rows.append({
            'Clasificación': cls,
            'Count': count,
            'Percent': pct,
            'AvgLength': avg_len,
            'Examples': examples
        })
    
    # Presentación
    for row in sorted(summary_rows, key=lambda x: -x['Count']):
        print(f"=== {row['Clasificación']} ===")
        print(f"  • Count: {row['Count']} ({row['Percent']:.1f}%)")
        print(f"  • Avg. length: {row['AvgLength']:.1f} tokens")
        print("  • Examples:")
        for ex in row['Examples']:
            print(f"      – {ex}")
        print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Genera estadísticas y ejemplos por clasificación de oraciones"
    )pyt
    parser.add_argument(
        "csv_file",
        help="Ruta al archivo CSV (e.g. clasificacion_resultados2.csv)"
    )
    parser.add_argument(
        "--examples", "-n",
        type=int, default=3,
        help="Número de ejemplos por clasificación"
    )
    args = parser.parse_args()
    
    summarize(args.csv_file, examples_per_class=args.examples)