import Link from 'next/link';

const FEATURE_CARDS = [
  {
    title: 'Training',
    description:
      'Submit a YOLO training job, configure hyperparameters, and watch live log output as it runs.',
    href: '/training',
    icon: '▶',
  },
  {
    title: 'Experiments',
    description:
      'Track YOLO training runs, compare metrics across runs, and view full experiment history.',
    href: '/experiments',
    icon: '⚗',
  },
  {
    title: 'Models',
    description:
      'Browse and manage registered models in the MLflow model registry by version and stage.',
    href: '/models',
    icon: '⬡',
  },
  {
    title: 'Inference',
    description:
      'Upload an image, run object detection against the deployed model, and visualize bounding boxes.',
    href: '/inference',
    icon: '◎',
  },
];

const SERVICES = [
  { label: 'MLflow', port: ':5000', role: 'Experiment tracking' },
  { label: 'MinIO', port: ':9000/9001', role: 'Artifact storage' },
  { label: 'Inference API', port: ':8000', role: 'Detection endpoint' },
  { label: 'PostgreSQL', port: ':5432', role: 'Metadata store' },
];

/** Home dashboard — platform overview with quick navigation cards. */
export default function HomePage() {
  return (
    <div>
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-100">Dashboard</h2>
        <p className="text-gray-500 mt-1">On-premise Vision MLOps Platform</p>
      </div>

      {/* Navigation cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        {FEATURE_CARDS.map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="group block p-6 rounded-xl bg-gray-900 border border-gray-800 hover:border-gray-600 transition-colors"
          >
            <span className="text-2xl">{card.icon}</span>
            <h3 className="mt-3 text-base font-semibold text-gray-100 group-hover:text-blue-400 transition-colors">
              {card.title}
            </h3>
            <p className="mt-2 text-sm text-gray-500 leading-relaxed">{card.description}</p>
          </Link>
        ))}
      </div>

      {/* Service stack info */}
      <div className="p-5 rounded-xl bg-gray-900 border border-gray-800">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Service Stack</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {SERVICES.map((svc) => (
            <div key={svc.label} className="px-3 py-2.5 rounded-lg bg-gray-800">
              <p className="text-xs font-medium text-gray-300">
                {svc.label}
                <span className="ml-1 text-gray-600">{svc.port}</span>
              </p>
              <p className="text-xs text-gray-600 mt-0.5">{svc.role}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
