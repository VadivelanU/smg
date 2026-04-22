use std::{collections::HashMap, sync::Arc};

use super::{
    AnthropicProvider, GeminiProvider, OpenAIProvider, Provider, SGLangProvider, XAIProvider,
};
use crate::worker::ProviderType;

pub struct ProviderRegistry {
    providers: HashMap<ProviderType, Arc<dyn Provider>>,
    default_provider: Arc<dyn Provider>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    pub fn new() -> Self {
        let mut providers = HashMap::new();

        providers.insert(
            ProviderType::OpenAI,
            Arc::new(OpenAIProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::XAI,
            Arc::new(XAIProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::Gemini,
            Arc::new(GeminiProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::Anthropic,
            Arc::new(AnthropicProvider) as Arc<dyn Provider>,
        );

        Self {
            providers,
            default_provider: Arc::new(SGLangProvider),
        }
    }

    pub fn get(&self, provider_type: &ProviderType) -> &dyn Provider {
        self.providers
            .get(provider_type)
            .map(|p| p.as_ref())
            .unwrap_or_else(|| self.default_provider.as_ref())
    }

    pub fn get_arc(&self, provider_type: &ProviderType) -> Arc<dyn Provider> {
        self.providers
            .get(provider_type)
            .cloned()
            .unwrap_or_else(|| Arc::clone(&self.default_provider))
    }

    #[expect(dead_code)]
    pub fn get_for_model(&self, model_name: &str) -> &dyn Provider {
        match ProviderType::from_model_name(model_name) {
            Some(pt) => self.get(&pt),
            None => self.default_provider.as_ref(),
        }
    }

    #[expect(dead_code)]
    pub fn default_provider(&self) -> &dyn Provider {
        self.default_provider.as_ref()
    }

    pub fn default_provider_arc(&self) -> Arc<dyn Provider> {
        Arc::clone(&self.default_provider)
    }
}
