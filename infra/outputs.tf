output "app_service_url" {
  value = "https://${azurerm_app_service.main.default_site_hostname}"
}
