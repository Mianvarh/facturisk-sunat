# Diccionario de datos

## Variables históricas

| Columna | Tipo detectado | Ejemplo | Origen | Tratamiento | Uso en modelo | Motivo de exclusión |
|---|---:|---|---|---|---|---|
| `Origen` | `string` | Borradores | Dataset histórico | Conservada para trazabilidad | No |  |
| `ID_Comprobante` | `string` | F003-04633228 | Dataset histórico | Conservada para trazabilidad | No |  |
| `RUC_Proveedor` | `string` | 20525994741 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Tipo_Comprobante` | `string` | Factura | Dataset histórico | Normalizada/derivada | Sí |  |
| `Fecha_Creacion` | `datetime64[ns]` | 2025-06-30 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Fecha_Emision` | `datetime64[ns]` | 2025-06-27 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Fecha_Vencimiento` | `datetime64[ns]` | 2025-07-27 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Fecha_Pago` | `datetime64[ns]` | 2025-08-02 | Dataset histórico | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Moneda` | `string` | No especificado | Dataset histórico | Normalizada/derivada | Sí |  |
| `Importe_Total` | `float64` | 8198.91 | Dataset histórico | Normalizada/derivada | Sí |  |
| `Estado` | `string` | Borrador | Dataset histórico | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Estado_Grupo` | `string` | Borrador | Dataset histórico | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Estado_Emision` | `string` | No emitida | Dataset histórico | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Estado_Aceptacion` | `string` | Pendiente / Borrador | Dataset histórico | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Estado_Pago` | `string` | Pagada | Dataset histórico | Conservada para trazabilidad | No |  |
| `Flag_Factura` | `Int8` | 1 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Flag_Emitida` | `Int8` | 0 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Flag_Aceptada` | `Int8` | 0 | Dataset histórico | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Flag_Observada` | `Int8` | 0 | Dataset histórico | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Flag_Rechazada` | `Int8` | 0 | Dataset histórico | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Flag_Pendiente_Pago` | `Int8` | 0 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Anio` | `Int32` | 2025 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Trimestre` | `string` | Q2 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Mes_Num` | `Int32` | 6 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Mes` | `string` | Junio | Dataset histórico | Conservada para trazabilidad | No |  |
| `Anio_Mes` | `string` | 2025-06 | Dataset histórico | Conservada para trazabilidad | No |  |
| `Orden_Anio_Mes` | `Int32` | 202506 | Dataset histórico | Conservada para trazabilidad | No |  |

## Dataset de entrenamiento

| Columna | Tipo detectado | Ejemplo | Origen | Tratamiento | Uso en modelo | Motivo de exclusión |
|---|---:|---|---|---|---|---|
| `Origen` | `str` | PreRegistros | Preparación | Conservada para trazabilidad | No |  |
| `ID_Comprobante` | `str` | F001-57022055 | Preparación | Conservada para trazabilidad | No |  |
| `RUC_Proveedor` | `int64` | 20493434340 | Preparación | Conservada para trazabilidad | No |  |
| `Tipo_Comprobante` | `str` | Factura | Preparación | Normalizada/derivada | Sí |  |
| `Fecha_Creacion` | `str` | 2025-08-02 | Preparación | Conservada para trazabilidad | No |  |
| `Fecha_Emision` | `str` | 2025-08-02 | Preparación | Conservada para trazabilidad | No |  |
| `Fecha_Vencimiento` | `str` | 2025-10-01 | Preparación | Conservada para trazabilidad | No |  |
| `Fecha_Pago` | `str` | 2025-10-01 | Preparación | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Moneda` | `str` | PEN | Preparación | Normalizada/derivada | Sí |  |
| `Importe_Total` | `float64` | 1749.85 | Preparación | Normalizada/derivada | Sí |  |
| `Estado` | `str` | Emitida | Preparación | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Estado_Grupo` | `str` | Emitida / Aceptada | Preparación | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Estado_Emision` | `str` | Emitida | Preparación | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Estado_Aceptacion` | `str` | Aceptada | Preparación | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Estado_Pago` | `str` | Pagada | Preparación | Conservada para trazabilidad | No |  |
| `Flag_Factura` | `int64` | 1 | Preparación | Conservada para trazabilidad | No |  |
| `Flag_Emitida` | `int64` | 1 | Preparación | Conservada para trazabilidad | No |  |
| `Flag_Aceptada` | `int64` | 1 | Preparación | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Flag_Observada` | `int64` | 0 | Preparación | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Flag_Rechazada` | `int64` | 0 | Preparación | Conservada para trazabilidad | No | Fuga de información o resultado conocido |
| `Flag_Pendiente_Pago` | `int64` | 0 | Preparación | Conservada para trazabilidad | No |  |
| `Anio` | `int64` | 2025 | Preparación | Conservada para trazabilidad | No |  |
| `Trimestre` | `str` | Q3 | Preparación | Conservada para trazabilidad | No |  |
| `Mes_Num` | `int64` | 8 | Preparación | Conservada para trazabilidad | No |  |
| `Mes` | `str` | Agosto | Preparación | Conservada para trazabilidad | No |  |
| `Anio_Mes` | `str` | 2025-08 | Preparación | Conservada para trazabilidad | No |  |
| `Orden_Anio_Mes` | `int64` | 202508 | Preparación | Conservada para trazabilidad | No |  |
| `RUC` | `int64` | 20493434340 | Preparación | Conservada para trazabilidad | No |  |
| `Razon_Social_SUNAT` | `str` |  | Preparación | Conservada para trazabilidad | No |  |
| `Estado_RUC` | `str` | ACTIVO | Preparación | Normalizada/derivada | Sí |  |
| `Condicion_Domicilio` | `str` | HABIDO | Preparación | Normalizada/derivada | Sí |  |
| `Ubigeo` | `int64` | 160101 | Preparación | Conservada para trazabilidad | No |  |
| `Domicilio_Fiscal` | `str` |  | Preparación | Conservada para trazabilidad | No |  |
| `Situacion_Tributaria_Actual` | `int64` | 0 | Preparación | Normalizada/derivada | Sí |  |
| `Fecha_Consulta` | `str` | 2026-09-25 | Preparación | Conservada para trazabilidad | No |  |
| `Fuente` | `str` | SUNAT - Padrón Reducido | Preparación | Conservada para trazabilidad | No |  |
| `SUNAT_Encontrado` | `int64` | 1 | Preparación | Normalizada/derivada | Sí |  |
| `plazo_dias` | `int64` | 60 | Preparación | Normalizada/derivada | Sí |  |
| `mes_emision` | `int64` | 8 | Preparación | Normalizada/derivada | Sí |  |
| `anio_emision` | `int64` | 2025 | Preparación | Normalizada/derivada | No |  |
| `trimestre_emision` | `int64` | 3 | Preparación | Normalizada/derivada | Sí |  |
| `dia_semana_emision` | `int64` | 5 | Preparación | Normalizada/derivada | Sí |  |
| `log_importe` | `float64` | 7.467856663282273 | Preparación | Normalizada/derivada | Sí |  |
| `Incidencia_Aceptacion` | `int64` | 0 | Preparación | Conservada para trazabilidad | No | Variable objetivo |
| `comprobantes_previos_proveedor` | `int64` | 1 | Preparación | Normalizada/derivada | Sí |  |
| `incidencias_previas_proveedor` | `float64` | 0.0 | Preparación | Normalizada/derivada | Sí |  |
| `tasa_incidencia_previa_proveedor` | `float64` | 0.0694488699124819 | Preparación | Normalizada/derivada | Sí |  |
| `importe_promedio_previo_proveedor` | `float64` | 7307.4479 | Preparación | Normalizada/derivada | Sí |  |
| `importe_maximo_previo_proveedor` | `float64` | 7307.4479 | Preparación | Normalizada/derivada | Sí |  |
| `dias_desde_comprobante_anterior` | `float64` | 623.0 | Preparación | Normalizada/derivada | Sí |  |
| `proveedor_nuevo` | `int64` | 0 | Preparación | Normalizada/derivada | Sí |  |
| `comprobantes_ultimos_30_dias` | `int64` | 0 | Preparación | Normalizada/derivada | Sí |  |
| `comprobantes_ultimos_90_dias` | `int64` | 0 | Preparación | Normalizada/derivada | Sí |  |
| `incidencias_ultimos_30_dias` | `int64` | 0 | Preparación | Normalizada/derivada | Sí |  |
| `incidencias_ultimos_90_dias` | `int64` | 0 | Preparación | Normalizada/derivada | Sí |  |
| `desviacion_importe_vs_promedio_proveedor` | `float64` | -5557.597900000001 | Preparación | Normalizada/derivada | Sí |  |

## Columnas excluidas por fuga

- `Estado_Aceptacion`: revela directa o indirectamente el resultado.
- `Flag_Aceptada`: revela directa o indirectamente el resultado.
- `Flag_Observada`: revela directa o indirectamente el resultado.
- `Flag_Rechazada`: revela directa o indirectamente el resultado.
- `Fecha_Pago`: revela directa o indirectamente el resultado.
- `Estado`: revela directa o indirectamente el resultado.
- `Estado_Grupo`: revela directa o indirectamente el resultado.
- `Estado_Emision`: revela directa o indirectamente el resultado.
- `Incidencia_Aceptacion`: revela directa o indirectamente el resultado.
