# Seguridad y credenciales

- Las credenciales de MongoDB se guardan localmente en `config/configuracion.json` o `.env`; ambos están excluidos de Git.
- La URI se muestra enmascarada en la aplicación y nunca se imprime en consola, logs ni reportes.
- Use un usuario de MongoDB con permisos solo sobre la base del proyecto.
- En MongoDB Atlas, autorice solo las IP necesarias en Network Access.
- Los datos importados (`data/raw/local/`) y los resultados (`outputs/`, `data/processed/`) no se versionan, porque pueden contener razones sociales y domicilios.
