output "id" {
  value = azurerm_search_service.rag.id
}

output "name" {
  value = azurerm_search_service.rag.name
}

output "endpoint" {
  value = "https://${azurerm_search_service.rag.name}.search.windows.net"
}
