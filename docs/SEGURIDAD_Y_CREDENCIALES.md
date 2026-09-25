# Seguridad y credenciales

La versión privada puede incluir una cuenta MongoDB de demostración en `config/configuracion.json`. Esa cuenta no debe ser administradora y debe limitarse a la base `facturisk`.

La contraseña no debe imprimirse en consola, logs, reportes ni documentación. Las credenciales no deben compartirse públicamente. Después de la prueba se recomienda cambiar o eliminar el usuario.

MongoDB Atlas puede requerir autorización de IP. Permitir `0.0.0.0/0` facilita pruebas, pero implica riesgo; es preferible una lista temporal y limitada.
