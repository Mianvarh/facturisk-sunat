# Errores frecuentes

- MongoDB no conecta: revisar internet, URI privada y lista de IP en Atlas.
- IP no autorizada: agregar temporalmente la IP del equipo en Network Access.
- SUNAT no responde: esperar o reutilizar la descarga válida existente.
- Dataset no encontrado: verificar `data/raw/comprobantes.parquet`.
- Error de formato: verificar la integridad de `data/raw/comprobantes.parquet`.
- Columnas faltantes: verificar el esquema de `data/raw/comprobantes.parquet`.
- Archivo bloqueado por Excel: cerrar Excel y volver a ejecutar.
- Modelo no encontrado: ejecutar entrenamiento o restaurar `data/models/modelo_incidencias.joblib`.
- Antivirus o SmartScreen bloquea el EXE: descomprimir en carpeta confiable y permitir ejecución.
- Rutas con espacios o tildes: el proyecto usa rutas portables, pero evite mover solo el EXE.
- Permisos de escritura: ejecutar desde una carpeta donde el usuario pueda escribir.
- Memoria insuficiente: el padrón SUNAT se procesa por chunks.
- PyInstaller hidden imports: revisar `build_windows_gui.spec`.
- EXE separado de `_internal`: asegurarse de mantener la estructura generada por PyInstaller.
