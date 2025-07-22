import pandas as pd
import plotly.express as px
import pdfplumber
import camelot
import re
import os

class ChartGenerator:
    def __init__(self):
        self.detected_tables = []
        self.dataframes = {}
    
    def extract_csv_tables(self, csv_file_path):
        try:
            df = pd.read_csv(csv_file_path)
            if not df.empty:
                table_id = f"csv_{os.path.basename(csv_file_path)}"
                self.dataframes[table_id] = df
                self.detected_tables.append({
                    'id': table_id,
                    'type': 'csv_file',
                    'dataframe': df,
                    'source': csv_file_path,
                    'description': f"CSV table ({len(df)} rows)"
                })
                print(f"✅ Extracted CSV table with {len(df)} rows from {csv_file_path}")
                return [table_id]
        except Exception as e:
            print(f"❌ CSV error: {e}")
        return []
    
    def extract_tables_from_pdf(self, pdf_path):
        tables_found = []
        
        try:
            print(f"Extracting tables from {pdf_path}...")
            # tables_found.extend(self._extract_with_pdfplumber(pdf_path))
            tables_found.extend(self._extract_with_camelot(pdf_path))
            print(f"Found {len(tables_found)} tables")
        except Exception as e:
            print(f"Error: {e}")
        
        self.detected_tables.extend(tables_found)
        return tables_found
    
    # def _extract_with_pdfplumber(self, pdf_path):
    #     tables = []
    #     with pdfplumber.open(pdf_path) as pdf:
    #         for page_num, page in enumerate(pdf.pages):
    #             for table_num, table_data in enumerate(page.extract_tables()):
    #                 if table_data and len(table_data) > 1:
    #                     df = self._clean_table_data(table_data)
    #                     if not df.empty:
    #                         table_id = f"pdf_p{page_num+1}_t{table_num+1}"
    #                         self.dataframes[table_id] = df
    #                         tables.append({
    #                             'id': table_id,
    #                             'type': 'pdf_table',
    #                             'dataframe': df,
    #                             'source': pdf_path,
    #                             'description': f"Table from page {page_num+1} ({len(df)} rows)"
    #                         })
    #     return tables
    
    def _extract_with_camelot(self, pdf_path):
        tables = []
        for method in ['lattice', 'stream']:
            try:
                camelot_tables = camelot.read_pdf(pdf_path, flavor=method, pages='all')
                for i, table in enumerate(camelot_tables):
                    if table.accuracy > 50:
                        df = self._clean_camelot_df(table.df)
                        if not df.empty and len(df.columns) > 1:
                            table_id = f"camelot_{method}_t{i+1}"
                            if not self._is_duplicate(df):
                                self.dataframes[table_id] = df
                                tables.append({
                                    'id': table_id,
                                    'type': f'camelot_{method}',
                                    'dataframe': df,
                                    'source': pdf_path,
                                    'description': f"{method.title()} table ({len(df)} rows, {table.accuracy:.0f}% accuracy)"
                                })
            except:
                continue
        return tables
    
    def _clean_table_data(self, table_data):
        cleaned = [[cell.strip() if cell else "" for cell in row] for row in table_data]
        cleaned = [row for row in cleaned if any(cell for cell in row)]
        
        if len(cleaned) < 2:
            return pd.DataFrame()
        
        headers = cleaned[0]
        data_rows = cleaned[1:]
        max_cols = len(headers)
        data_rows = [row[:max_cols] + [''] * (max_cols - len(row)) for row in data_rows]
        
        df = pd.DataFrame(data_rows, columns=headers)
        return df.loc[:, (df != '').any(axis=0)]
    
    def _clean_camelot_df(self, df):
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df = df[~(df == '').all(axis=1)].loc[:, ~(df == '').all()]
        df = df.reset_index(drop=True)
        
        if len(df) > 1 and not df.iloc[0].isnull().all():
            df.columns = df.iloc[0]
            df = df.drop(df.index[0]).reset_index(drop=True)
        
        return df
    
    def _is_duplicate(self, new_df):
        for existing_df in self.dataframes.values():
            if new_df.shape == existing_df.shape and len(new_df.columns) == len(existing_df.columns):
                return True
        return False
    
    def extract_tables_from_documents(self, documents):
        tables_found = []
        for i, doc_chunk in enumerate(documents):
            tables_found.extend(self._detect_patterns(doc_chunk, f"chunk_{i}"))
        self.detected_tables.extend(tables_found)
        return tables_found
    
    def _detect_patterns(self, text, source_id):
        tables = []
        lines = text.split('\n')
        
        pipe_table = self._extract_pipe_table(lines, source_id)
        if pipe_table:
            tables.append(pipe_table)
        
        tab_lines = [line.split('\t') for line in lines if '\t' in line and line.count('\t') >= 1]
        tab_lines = [[cell.strip() for cell in cells if cell.strip()] for cells in tab_lines]
        tab_lines = [cells for cells in tab_lines if len(cells) >= 2]
        
        if len(tab_lines) >= 2 and len(set(len(row) for row in tab_lines)) == 1:
            df = pd.DataFrame(tab_lines[1:], columns=tab_lines[0])
            table_id = f"{source_id}_tab"
            self.dataframes[table_id] = df
            tables.append({
                'id': table_id,
                'type': 'tab_separated',
                'dataframe': df,
                'source': source_id,
                'description': f"Tab table ({len(df)} rows)"
            })
        
        csv_lines = []
        for line in lines:
            if ',' in line and line.count(',') >= 1:
                cells = [cell.strip().strip('"\'') for cell in line.split(',')]
                if len(cells) >= 2:
                    csv_lines.append(cells)
        
        if len(csv_lines) >= 2:
            col_counts = [len(row) for row in csv_lines]
            target_cols = max(set(col_counts), key=col_counts.count)
            filtered_lines = [row for row in csv_lines if len(row) == target_cols]
            
            if len(filtered_lines) >= 2:
                df = pd.DataFrame(filtered_lines[1:], columns=filtered_lines[0])
                table_id = f"{source_id}_csv"
                self.dataframes[table_id] = df
                tables.append({
                    'id': table_id,
                    'type': 'csv_like',
                    'dataframe': df,
                    'source': source_id,
                    'description': f"CSV table ({len(df)} rows)"
                })
        
        return tables
    
    def _extract_pipe_table(self, lines, source_id):
        table_lines = []
        for line in lines:
            if '|' in line and line.count('|') >= 2:
                clean_line = line.strip().strip('|')
                if clean_line:
                    table_lines.append([cell.strip() for cell in clean_line.split('|')])
            elif table_lines:
                break
        
        if len(table_lines) >= 2:
            cleaned_lines = [line_data for line_data in table_lines 
                           if not all(re.match(r'^[-\s:]*$', cell) for cell in line_data)]
            
            if len(cleaned_lines) >= 2:
                df = pd.DataFrame(cleaned_lines[1:], columns=cleaned_lines[0])
                table_id = f"{source_id}_pipe"
                self.dataframes[table_id] = df
                return {
                    'id': table_id,
                    'type': 'pipe_separated',
                    'dataframe': df,
                    'source': source_id,
                    'description': f"Pipe table ({len(df)} rows)"
                }
        return None
    
    def extract_excel_tables(self, excel_file_path):
        try:
            excel_data = pd.read_excel(excel_file_path, sheet_name=None)
            for sheet_name, df in excel_data.items():
                if not df.empty:
                    table_id = f"excel_{sheet_name}"
                    self.dataframes[table_id] = df
                    self.detected_tables.append({
                        'id': table_id,
                        'type': 'excel_sheet',
                        'dataframe': df,
                        'source': excel_file_path,
                        'description': f"Excel '{sheet_name}' ({len(df)} rows)"
                    })
            return list(excel_data.keys())
        except Exception as e:
            print(f"Excel error: {e}")
            return []

    def generate_chart(self, table_id, chart_type, x_column=None, y_column=None, color_column=None):
        if table_id not in self.dataframes:
            return None, "Table not found"
        
        df = self.dataframes[table_id]
        
        try:
            numeric_df = self._convert_to_numeric(df)
            
            chart_map = {
                "bar": lambda: px.bar(numeric_df, x=x_column, y=y_column, color=color_column),
                "line": lambda: px.line(numeric_df, x=x_column, y=y_column, color=color_column),
                "scatter": lambda: px.scatter(numeric_df, x=x_column, y=y_column, color=color_column),
                "pie": lambda: px.pie(values=df[x_column].value_counts().values, 
                                    names=df[x_column].value_counts().index) if x_column else None,
                "histogram": lambda: px.histogram(numeric_df, x=x_column),
                "box": lambda: px.box(numeric_df, y=y_column, x=x_column)
            }
            
            if chart_type not in chart_map:
                return None, f"Unsupported chart: {chart_type}"
            
            fig = chart_map[chart_type]()
            if fig is None:
                return None, "Missing required column"
            
            fig.update_layout(height=500, title=f"{chart_type.title()} Chart")
            return fig, "Chart created"
            
        except Exception as e:
            return None, f"Chart error: {str(e)}"
    
    def _convert_to_numeric(self, df):
        numeric_df = df.copy()
        for col in numeric_df.columns:
            try:
                cleaned = numeric_df[col].astype(str).str.replace(r'[,$%\s]+', '', regex=True)
                numeric_df[col] = pd.to_numeric(cleaned, errors='ignore')
            except:
                pass
        return numeric_df
    
    def get_table_info(self, table_id):
        if table_id not in self.dataframes:
            return None
        
        df = self.dataframes[table_id]
        df_numeric = self._convert_to_numeric(df)
        
        numeric_cols = []
        categorical_cols = []
        
        for col in df_numeric.columns:
            if pd.api.types.is_numeric_dtype(df_numeric[col]) or \
            df_numeric[col].astype(str).str.match(r'^-?\d*\.?\d+$').all():
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'sample_data': df.head().to_dict('records'),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols
        }
    
    def suggest_charts(self, table_id):
        info = self.get_table_info(table_id)
        if not info:
            return []
        
        numeric_cols = info['numeric_columns']
        categorical_cols = info['categorical_columns']
        suggestions = []
        
        chart_rules = [
            (len(numeric_cols) >= 1 and len(categorical_cols) >= 1, [
                {'type': 'bar', 'description': 'Compare values by category', 'x_options': categorical_cols, 'y_options': numeric_cols},
                {'type': 'box', 'description': 'Show value distribution', 'x_options': categorical_cols, 'y_options': numeric_cols}
            ]),
            (len(numeric_cols) >= 2, [
                {'type': 'scatter', 'description': 'Show relationships', 'x_options': numeric_cols, 'y_options': numeric_cols},
                {'type': 'line', 'description': 'Show trends', 'x_options': info['columns'], 'y_options': numeric_cols}
            ]),
            (len(numeric_cols) >= 1, [
                {'type': 'histogram', 'description': 'Show distribution', 'x_options': numeric_cols, 'y_options': []}
            ]),
            (len(categorical_cols) >= 1, [
                {'type': 'pie', 'description': 'Show category distribution', 'x_options': categorical_cols, 'y_options': []}
            ])
        ]
        
        for condition, charts in chart_rules:
            if condition:
                suggestions.extend(charts)
        
        print(f"Table {table_id} has numeric columns: {numeric_cols}")
        print(f"Table {table_id} has categorical columns: {categorical_cols}")
        print(f"Suggested {len(suggestions)} charts")
    
        return suggestions
    
    def get_all_tables(self):
        return self.detected_tables
    
    def clear_tables(self):
        self.detected_tables = []
        self.dataframes = {}