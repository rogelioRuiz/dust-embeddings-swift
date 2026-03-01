Pod::Spec.new do |s|
  s.name = 'DustEmbeddings'
  s.version = File.read(File.join(__dir__, 'VERSION')).strip
  s.summary = 'Standalone tokenizers and embedding runtime primitives for Dust.'
  s.license = { :type => 'Apache-2.0', :file => 'LICENSE' }
  s.homepage = 'https://github.com/rogelioRuiz/dust-embeddings-swift'
  s.author = 'Techxagon'
  s.source = { :git => 'https://github.com/rogelioRuiz/dust-embeddings-swift.git', :tag => s.version.to_s }

  s.source_files = 'Sources/DustEmbeddings/**/*.swift'
  s.module_name = 'DustEmbeddings'
  s.ios.deployment_target = '16.0'
  s.swift_version = '5.9'

  s.dependency 'DustCore'
  s.dependency 'DustOnnx'
end
