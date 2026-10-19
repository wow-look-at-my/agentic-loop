package commonai

// oaExtraFields is the extra_fields object a Bifrost gateway adds to every response and every stream chunk.
type oaExtraFields struct {
	RoutingInfo *struct {
		Provider string `json:"provider"`
		Model    string `json:"model"`
	} `json:"routing_info"`
}

// routed is the model-list id the gateway sent the request to. Its list names a provider's model as provider/model.
func (f *oaExtraFields) routed() string {
	if f == nil || f.RoutingInfo == nil || f.RoutingInfo.Model == "" {
		return ""
	}
	if f.RoutingInfo.Provider == "" {
		return f.RoutingInfo.Model
	}
	return f.RoutingInfo.Provider + "/" + f.RoutingInfo.Model
}
